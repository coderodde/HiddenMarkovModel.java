package com.github.coderodde.hmm.demo;

import com.github.coderodde.hmm.HiddenMarkovModel;
import com.github.coderodde.hmm.HiddenMarkovModelState;
import com.github.coderodde.hmm.HiddenMarkovModelStatePath;
import com.github.coderodde.hmm.HiddenMarkovModelStateType;
import java.util.List;
import java.util.Random;

/**
 * This class implements the demonstration of the hidden Markov model.
 */
public final class HMMDemo {
    
    public static void main(String[] args) {
        
        Random random = new Random(13L);
        
        HiddenMarkovModelState startState = 
                new HiddenMarkovModelState(0, HiddenMarkovModelStateType.START);
        
        HiddenMarkovModelState endState = 
                new HiddenMarkovModelState(3, HiddenMarkovModelStateType.END);
        
        HiddenMarkovModelState codingState = 
                new HiddenMarkovModelState(
                        1,
                        HiddenMarkovModelStateType.HIDDEN);
        
        HiddenMarkovModelState noncodingState = 
                new HiddenMarkovModelState(
                        2,
                        HiddenMarkovModelStateType.HIDDEN);
        
        HiddenMarkovModel hmm = new HiddenMarkovModel(startState, 
                                                      endState,
                                                      random);
        
        // BEGIN: State transitions.
        startState.addStateTransition(noncodingState, 0.49);
        startState.addStateTransition(codingState, 0.49);
        startState.addStateTransition(endState, 0.02);
        
        codingState.addStateTransition(codingState, 0.4);
        codingState.addStateTransition(endState, 0.3);
        codingState.addStateTransition(noncodingState, 0.3);
        
        noncodingState.addStateTransition(noncodingState, 0.3);
        noncodingState.addStateTransition(codingState, 0.35);
        noncodingState.addStateTransition(endState, 0.35);
        // END: State transitions.
        
        // BEGIN: Emissions.
        codingState.addEmissionTransition('A', 0.18);
        codingState.addEmissionTransition('C', 0.32);
        codingState.addEmissionTransition('G', 0.32);
        codingState.addEmissionTransition('T', 0.18);
        
        noncodingState.addEmissionTransition('A', 0.25);
        noncodingState.addEmissionTransition('C', 0.25);
        noncodingState.addEmissionTransition('G', 0.25);
        noncodingState.addEmissionTransition('T', 0.25);
        // END: Emissions.
        
        startState.normalize();
        codingState.normalize();
        noncodingState.normalize();
        endState.normalize();
        
        System.out.println("--- Composing random walks ---");
        
        for (int i = 0; i < 10; i++) {
            int lineNumber = i + 1;
            System.out.printf("%2d: %s\n", lineNumber, hmm.compose());
        }
        
        String sequence = "AGCG";
        
        List<HiddenMarkovModelStatePath> statePathSequences = 
                hmm.computeAllStatePaths(sequence);
        
        System.out.printf("Brute-force path inference for sequence \"%s\", " + 
                          "total probability = %f.\n",
                          sequence,
                          HiddenMarkovModel.sumPathProbabilities(
                                  statePathSequences));
        
        double hmmProbabilitySum = hmm.runForwardAlgorithm(sequence);
        
        System.out.printf("HMM total probability: %f.\n", hmmProbabilitySum);
        System.out.println("Brute-force state paths:");
        
        int lineNumber = 1;
        
        for (HiddenMarkovModelStatePath stateSequence : statePathSequences) {
            System.out.printf("%4d: %s\n", lineNumber++, stateSequence);
        }
    }
}
