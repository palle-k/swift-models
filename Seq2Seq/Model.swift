// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import TensorFlow


struct EncoderOutput: Differentiable {
    var outputSequence: Tensor<Float>
    var finalForwardState: LSTMCell<Float>.State
    var finalBackwardState: LSTMCell<Float>.State
}


struct Encoder: Module {
    var embedding: Embedding<Float>
    var forwardRNN: LSTM<Float>
    var backwardRNN: LSTM<Float>
    
    init(vocabularySize: Int, embeddingSize: Int, hiddenSize: Int) {
        self.embedding = Embedding(vocabularySize: vocabularySize, embeddingSize: embeddingSize)
        self.forwardRNN = LSTM(LSTMCell(inputSize: embeddingSize, hiddenSize: hiddenSize / 2))
        self.backwardRNN = LSTM(LSTMCell(inputSize: embeddingSize, hiddenSize: hiddenSize / 2))
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Int32>) -> Tensor<Float> {
        // input: [seqlen]
        let embedded = embedding(input).expandingShape(at: 1) // [seqlen, 1, emb dim]
        let elements = embedded.unstacked() // [[1, emb dim]]
        let forwardStates = self.forwardRNN(elements) // [[1, hidden dim / 2] * 2]
        let backwardStates = self.backwardRNN(elements)
        
        let forwardHidden = Tensor(stacking: forwardStates.differentiableMap {$0.hidden}) // [seqlen, 1, hidden dim / 2]
        let backwardHidden = Tensor(stacking: backwardStates.differentiableMap {$0.hidden})
        
        let outputSequence = Tensor(stacking: [forwardHidden, backwardHidden], alongAxis: 2) // [seqlen, 1, hidden dim]
        
        return outputSequence
    }
}


struct AttentionInput: Differentiable {
    var encoderStates: Tensor<Float>
    var decoderState: Tensor<Float>
    var attentionHistory: Tensor<Float>
}

struct AttentionOutput: Differentiable {
    var cumulativeScoreLogits: Tensor<Float>
    var normalizedScores: Tensor<Float>
}

struct TanhAttention: Module {
    var W_h: Tensor<Float>
    var W_s: Tensor<Float>
    var b: Tensor<Float>
    var v: Tensor<Float>
    
    @noDerivative let encoderHiddenSize: Int
    @noDerivative let decoderHiddenSize: Int
    @noDerivative let latentSize: Int
    @noDerivative let isTemporal: Bool
    
    /// Tanh attention (optionally intra-temporal)
    init(encoderHiddenSize: Int, decoderHiddenSize: Int, latentSize: Int, temporal: Bool) {
        self.encoderHiddenSize = encoderHiddenSize
        self.decoderHiddenSize = decoderHiddenSize
        self.latentSize = latentSize
        self.isTemporal = temporal
        
        W_h = Tensor(glorotNormal: [decoderHiddenSize, latentSize])
        W_s = Tensor(glorotNormal: [encoderHiddenSize, latentSize])
        b = Tensor(glorotNormal: [latentSize])
        v = Tensor(glorotNormal: [latentSize])
    }

    @differentiable(wrt: tensor)
    func normalize<Element: TensorFlowFloatingPoint>(_ tensor: Tensor<Element>, along axis: Int) -> Tensor<Element> {
        tensor / tensor.sum(alongAxes: axis).expandingShape(at: axis)
    }
    
    @differentiable
    func callAsFunction(_ input: AttentionInput) -> AttentionOutput {
        let encoderStateSequence = input.encoderStates // [seqlen, batchSize, encHS]
        let decoderState = input.decoderState // [batchSize, decHS]
        
        // let seqlen = encoderStateSequence.shape[0]
        let batchSize = decoderState.shape[0]
        
        let encIn = encoderStateSequence.reshaped(to: [-1, encoderHiddenSize])
        let encScore = matmul(encIn, W_s).reshaped(to: [-1, batchSize, latentSize]) // [seqlen * batchSize, latentSize]
        let decScore = matmul(decoderState, W_h) // [batchSize, latentSize]
        
        let scores = matmul(tanh(encScore + decScore + b) // [seqlen, batchSize, latentSize]
            .reshaped(to: [-1, latentSize]), // [seqlen * batchSize, latentSize]
            v.reshaped(to: [-1, 1])) // [seqlen * batchSize, 1]
            .reshaped(to: [-1, batchSize]) // [seqlen, batchSize]
        
        if !isTemporal  {
            return AttentionOutput(cumulativeScoreLogits: scores, normalizedScores: softmax(scores, alongAxis: 0))
        }
        
        let expScores = exp(scores) // [seqlen, batchSize]
        let attnHistory = input.attentionHistory // [seqlen, batchSize]
        
        if attnHistory.shape == [] {
            return AttentionOutput(cumulativeScoreLogits: scores, normalizedScores: softmax(scores, alongAxis: 0))
        }
        
        return AttentionOutput(cumulativeScoreLogits: expScores + attnHistory, normalizedScores: normalize(expScores / attnHistory, along: 0))
    }
}

struct AttentionCombineInput: Differentiable {
    var scores: Tensor<Float>
    var states: Tensor<Float>
}

struct AttentionCombine: ParameterlessLayer {
    @differentiable
    func callAsFunction(_ inputs: AttentionCombineInput) -> Tensor<Float> {
        let scores = inputs.scores // [seqlen, batchsize]
        let weights = inputs.states // [seqlen, batchSize, hiddenSize]
        
        let scoredWeights = (scores.expandingShape(at: 2) * weights).sum(alongAxes: 0) // [batchSize, hiddenSize]
        
        return scoredWeights
    }
}

struct DecoderInput: Differentiable {
    @noDerivative var token: Tensor<Int32>
    var previousState: LSTMCell<Float>.State
    var encoderStates: Tensor<Float>
    var cumulativeAttentionLogits: Tensor<Float>
}

struct DecoderOutput: Differentiable {
    var state: LSTMCell<Float>.State
    var logits: Tensor<Float>
    var cumulativeAttentionLogits: Tensor<Float>
}

struct Decoder: Layer {
    var embedding: Embedding<Float>
    var attention: TanhAttention
    var combine: AttentionCombine
    var cell: LSTMCell<Float>
    var output: Dense<Float>
    
    init(vocabularySize: Int, embeddingSize: Int, encoderHiddenSize: Int, hiddenSize: Int, useIntraTemporalAttention: Bool = true) {
        embedding = Embedding(vocabularySize: vocabularySize, embeddingSize: embeddingSize)
        attention = TanhAttention(encoderHiddenSize: encoderHiddenSize, decoderHiddenSize: hiddenSize, latentSize: hiddenSize, temporal: useIntraTemporalAttention)
        combine = AttentionCombine()
        cell = LSTMCell(inputSize: embeddingSize + encoderHiddenSize, hiddenSize: hiddenSize)
        output = Dense(inputSize: hiddenSize, outputSize: vocabularySize)
    }
    
    @differentiable
    func callAsFunction(_ input: DecoderInput) -> DecoderOutput {
        let embedded = embedding(input.token.reshaped(to: [1])) // [1, embedding size]
        
        let attnInput = AttentionInput(encoderStates: input.encoderStates, decoderState: input.previousState.hidden, attentionHistory: input.cumulativeAttentionLogits)
        let attnResult = attention(attnInput)
        
        let attendedStates = combine(AttentionCombineInput(scores: attnResult.normalizedScores, states: input.encoderStates)) // [1, enc hidden size]
        
        let rnnInput = Tensor(stacking: [embedded, attendedStates], alongAxis: 1)
        let rnnResult = cell(input: rnnInput, state: input.previousState) // [1, dec hidden size]
        let (rnnOutput, rnnState) = (rnnResult.output, rnnResult.state)
        
        let vocabLogits = output(rnnOutput.hidden) // [1, vocab size]
        
        return DecoderOutput(state: rnnState, logits: vocabLogits, cumulativeAttentionLogits: attnResult.cumulativeScoreLogits)
    }
}

struct Seq2SeqInput {
    var expectedSequence: Tensor<Int32>?
    var inputSequence: Tensor<Int32>
    var compilerCrashWorkaroundStartOutputLogits: Tensor<Float>
    var maxLength: Int
    var startToken: Int32
    var endToken: Int32
}

struct Seq2Seq: Module {
    var encoder: Encoder
    var mediator: Dense<Float>
    var decoder: Decoder
    
    init(sourceVocabSize: Int, destinationVocabSize: Int, sourceEmbeddingSize: Int, destinationEmbeddingSize: Int, encoderHiddenSize: Int, decoderHiddenSize: Int, useIntraTemporalAttention: Bool) {
        self.encoder = Encoder(
            vocabularySize: sourceVocabSize,
            embeddingSize: sourceEmbeddingSize,
            hiddenSize: encoderHiddenSize
        )
        self.mediator = Dense(inputSize: encoderHiddenSize, outputSize: 2 * decoderHiddenSize, activation: relu)
        self.decoder = Decoder(
            vocabularySize: destinationVocabSize,
            embeddingSize: destinationEmbeddingSize,
            encoderHiddenSize: encoderHiddenSize,
            hiddenSize: decoderHiddenSize,
            useIntraTemporalAttention: useIntraTemporalAttention
        )
    }
    
    @differentiable
    func callAsFunction(_ input: Seq2SeqInput) -> Tensor<Float> {
        let encoded = encoder(input.inputSequence)
        let finalEncoderState = encoded[-1]  // [batch size (1), enc hidden size]
        let decoderInput = mediator(finalEncoderState).reshaped(to: [finalEncoderState.shape[0], 2, -1])
        let decoderInputState = LSTMCell<Float>.State(
            cell: decoderInput.slice(lowerBounds: [0, 0, 0], upperBounds: [1, 1, -1]),
            hidden: decoderInput.slice(lowerBounds: [0, 1, 0], upperBounds: [1, 2, -1])
        )

        var currentDecoderState = decoderInputState
        var cumulativeAttnLogits = Tensor<Float>(zeros: [])

        var outputLogits: Tensor<Float> = input.compilerCrashWorkaroundStartOutputLogits

        if input.expectedSequence != nil /* [seqlen, 1] */ {
            let expectedSequence = input.expectedSequence!
//            // teacher forcing
            for t in 0 ..< (expectedSequence.shape[0] - 1) {
                let decoderInput = DecoderInput(
                    token: expectedSequence[t + 1],
                    previousState: currentDecoderState,
                    encoderStates: encoded,
                    cumulativeAttentionLogits: cumulativeAttnLogits
                )
                let output = decoder(decoderInput)
                currentDecoderState = output.state
                cumulativeAttnLogits = output.cumulativeAttentionLogits

                if outputLogits.shape != [] {
                    outputLogits = Tensor(stacking: [outputLogits, output.logits.reshaped(to: [1, 1, -1])], alongAxis: 0)
                } else {
                    outputLogits = output.logits.reshaped(to: [1, 1, -1])
                }
            }
        } else {
//            // no teacher forcing
            var previousToken = Tensor([input.startToken])

            for t in 0 ..< input.maxLength {
                let decoderInput = DecoderInput(
                    token: previousToken,
                    previousState: currentDecoderState,
                    encoderStates: encoded,
                    cumulativeAttentionLogits: cumulativeAttnLogits
                )
                let output = decoder(decoderInput)
                currentDecoderState = output.state
                cumulativeAttnLogits = output.cumulativeAttentionLogits
                previousToken = output.logits.flattened().argmax()

                if outputLogits.shape != [] {
                    outputLogits = Tensor(stacking: [outputLogits, output.logits.reshaped(to: [1, 1, -1])], alongAxis: 0)
                } else {
                    outputLogits = output.logits.reshaped(to: [1, 1, -1])
                }

                if previousToken.scalars[0] == input.endToken {
                    break
                }
            }
        }

        return outputLogits
    }
}
