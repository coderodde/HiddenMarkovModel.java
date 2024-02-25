package com.github.coderodde.hmm;

import static java.lang.Math.E;
import static java.lang.Math.log;
import static java.lang.Math.pow;
import java.util.List;

/**
 * This class stores the state path over the hidden states of an HMM.
 */
public final class HiddenMarkovModelStatePath 
    implements Comparable<HiddenMarkovModelStatePath> {
    
    private final List<HiddenMarkovModelState> stateSequence;
    private final double probability;
    
    HiddenMarkovModelStatePath(List<HiddenMarkovModelState> stateSequence,
                                   String observedSymbols,
                                   HiddenMarkovModel hiddenMarkovModel) {
        
        this.stateSequence = stateSequence;
        this.probability = computeJointProbability(observedSymbols);
    }
    
    public int size() {
        return stateSequence.size();
    }
    
    public HiddenMarkovModelState getState(int stateIndex) {
        return stateSequence.get(stateIndex);
    }
    
    public double getProbability() {
        return probability;
    }
    
    @Override
    public int compareTo(HiddenMarkovModelStatePath o) {
        return Double.compare(probability, o.probability);
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        
        boolean first = true;
        
        for (HiddenMarkovModelState state : stateSequence) {
            if (first) {
                first = false;
            } else {
                sb.append(", ");
            }
            
            sb.append(state.getId());
        }
        
        sb.append("| p = ");
        sb.append(probability);
        sb.append("]");
        return sb.toString();
    }
    
    /**
     * Computes the joint probability of this path.
     * 
     * @param observedSymbols the observation text.
     * 
     * @return the joint probability of this path and the input text.
     */
    private double computeJointProbability(String observedSymbols) {
        double logProbability = computeEmissionProbabilities(observedSymbols) +
                                computeTransitionProbabilities();
        
        // Convert to probability:
        return pow(E, logProbability);
    }
    
    /**
     * Computes the product of emission probabilities over the input text.
     * 
     * @param observedSymbols the input text.
     * 
     * @return the total emission probability.
     */
    private double computeEmissionProbabilities(String observedSymbols) {
        double probability = 0.0;
        
        for (int i = 0; i != observedSymbols.length(); i++) {
            char observedSymbol = observedSymbols.charAt(i);
            HiddenMarkovModelState state = stateSequence.get(i + 1);
            probability += log(state.getEmissions().get(observedSymbol));
        }
        
        return probability;
    }
    
    /**
     * Computes the product of transition probabilities over this path.
     * 
     * @return the product of transitions.
     */
    private double computeTransitionProbabilities() {
        double probability = 0.0;
        
        for (int i = 0; i < stateSequence.size() - 1; i++) {
            HiddenMarkovModelState sourceState = stateSequence.get(i);
            HiddenMarkovModelState targetState = stateSequence.get(i + 1);
            probability += log(sourceState.getFollowingStates()
                                          .get(targetState));
        }
        
        return probability;
    }
}
