    package com.github.coderodde.hmm;

    import java.util.ArrayDeque;
    import java.util.ArrayList;
    import java.util.Arrays;
    import java.util.Collections;
    import java.util.Deque;
    import java.util.HashMap;
    import java.util.HashSet;
    import java.util.List;
    import java.util.Map;
    import java.util.Random;
    import java.util.Set;

    /**
     * This class implements an HMM (hidden Markov model).
     */
    public final class HiddenMarkovModel {

        /**
         * Used to denote the Viterbi matrix cells that are not yet computed.
         */
        private static final double UNSET_PROBABILITY = -1.0;

        /**
         * The start state of the process.
         */
        private final HiddenMarkovModelState startState;

        /**
         * The end state of the process.
         */
        private final HiddenMarkovModelState endState;

        private final Random random;

        public HiddenMarkovModel(HiddenMarkovModelState startState,
                                 HiddenMarkovModelState endState,
                                 Random random) {

            this.startState = startState;
            this.endState = endState;
            this.random = random;
        }

        public static double sumPathProbabilities(
                List<HiddenMarkovModelStatePath> paths) {

            double psum = 0.0;

            for (HiddenMarkovModelStatePath path : paths) {
                psum += path.getProbability();
            }

            return psum;
        }

        /**
         * Performs a brute-force computation of the list of all possible paths in
         * this HMM.
         * 
         * @param sequence the target text.
         * @return the list of sequences, sorted from most probable to the most
         *         improbable.
         */
        public List<HiddenMarkovModelStatePath> 
            computeAllStatePaths(String sequence) {

            int expectedStatePathSize = sequence.length() + 2;

            List<List<HiddenMarkovModelState>> statePaths = new ArrayList<>();
            List<HiddenMarkovModelState> currentPath = new ArrayList<>();

            // BEGIN: Do the search:
            currentPath.add(startState);

            depthFirstSearchImpl(statePaths, 
                                 currentPath,
                                 expectedStatePathSize, 
                                 startState);

            // END: Searching done.

            // Construct the sequences:
            List<HiddenMarkovModelStatePath> sequenceList = 
                    new ArrayList<>(statePaths.size());

            for (List<HiddenMarkovModelState> statePath : statePaths) {
                sequenceList.add(
                        new HiddenMarkovModelStatePath(
                                statePath, 
                                sequence,
                                this));
            }

            // Put into descending order by probabilities:
            Collections.sort(sequenceList);
            Collections.reverse(sequenceList);

            return sequenceList;
        }

        /**
         * Returns the most probable state path for the input sequence using the 
         * <a href="https://en.wikipedia.org/wiki/Viterbi_algorithm">
         * Viterbi algorithm.</a>
         * 
         * @param sequence the target sequence.
         * @return the state path.
         */
        public HiddenMarkovModelStatePath runViterbiAlgorithm(String sequence) {

            // Get all the states reachable from the start state:
            Set<HiddenMarkovModelState> graph = computeAllStates();

            if (!graph.contains(endState)) {
                // End state unreachable. Abort.
                throw new IllegalStateException("End state is unreachable.");
            }

            // Maps the column index to the corresponding state:
            Map<Integer, HiddenMarkovModelState> stateMap = 
                    new HashMap<>(graph.size());

            // Maps the state to the corresponding column index:
            Map<HiddenMarkovModelState, Integer> inverseStateMap = 
                    new HashMap<>(graph.size());

            loadStateMaps(graph, 
                          stateMap, 
                          inverseStateMap);
            
            // Computes the entire Viterbi matrix:
            double[][] viterbiMatrix =
                    computeViterbiMatrix(
                            sequence,
                            stateMap,
                            inverseStateMap);

            // Calls the dynamic programming result reconstruction:
            return tracebackStateSequenceViterbi(viterbiMatrix,
                                          sequence,
                                          stateMap,
                                          inverseStateMap);
        }

        /**
         * Returns the sum of probabilities over all the feasible paths that accord
         * to the {@code sequence}.
         * 
         * @param sequence the sequence text.
         * 
         * @return the sum of probabilities.
         */
        public double runForwardAlgorithm(String sequence) {

            // Get all the states reachable from the start state:
            Set<HiddenMarkovModelState> graph = computeAllStates();

            if (!graph.contains(endState)) {
                // End state unreachable. Abort.
                throw new IllegalStateException("End state is unreachable.");
            }
            
            Map<Integer, HiddenMarkovModelState> stateMap =
                    new HashMap<>(graph.size());
            
            Map<HiddenMarkovModelState, Integer> inverseStateMap = 
                    new HashMap<>(graph.size());
            
            loadStateMaps(graph, 
                          stateMap,
                          inverseStateMap);

            // Computes the entire forward matrix:
            double[][] forwardMatrix =
                    computeForwardMatrix(
                            sequence,
                            stateMap,
                            inverseStateMap);

            // Uses the dynamic programming result reconstruction:
            return tracebackStateSequenceForward(forwardMatrix, 
                                                 inverseStateMap);
        }

        /**
         * Composes a sequence according to this HMM.
         * 
         * @return a sequence.
         */
        public String compose() {
            StringBuilder sb = new StringBuilder();

            HiddenMarkovModelState currentState = startState;

            while (true) {
                currentState = doStateTransition(currentState);

                if (currentState.equals(endState)) {
                    // Once here, we are done:
                    return sb.toString();
                }

                sb.append(doEmit(currentState));
            }
        }

        /**
         * Computes the entire Viterbi matrix.
         * 
         * @param sequence        the input text.
         * @param stateMap        the state map. From column index to the state.
         * @param inverseStateMap the inverse state map. From state to column index.
         * 
         * @return the entire Viterbi matrix.
         */
        private double[][] computeViterbiMatrix(
                String sequence,
                Map<Integer, HiddenMarkovModelState> stateMap,
                Map<HiddenMarkovModelState, Integer> inverseStateMap) {

            double[][] matrix = new double[sequence.length() + 1]
                                          [stateMap.size()];

            // Set all required cells to unset sentinel:
            for (int rowIndex = 1; rowIndex < matrix.length; rowIndex++) {
                Arrays.fill(matrix[rowIndex], UNSET_PROBABILITY);
            }

            // BEGIN: Base case initialization.
            matrix[0][0] = 1.0;

            for (int columnIndex = 1; 
                     columnIndex < matrix[0].length;
                     columnIndex++) {

                matrix[0][columnIndex] = 0.0;
            }
            // END: Done with the base case initialization.

            // Recursively build the matrix:
            for (int h = 1; h < matrix[0].length - 1; h++) {
                matrix[sequence.length()][h] = 
                    computeViterbiMatrixImpl(
                            sequence.length(),
                            h,
                            matrix, 
                            sequence, 
                            stateMap,
                            inverseStateMap);
            }

            return matrix;
        }

        /**
         * Computes the entire forward matrix.
         * 
         * @param sequence        the input text.
         * @param stateMap        the state map. From column index to the state.
         * @param inverseStateMap the inverse state map. From state to column index.
         * 
         * @return the entire Viterbi matrix.
         */
        private double[][] computeForwardMatrix(
                String sequence,
                Map<Integer, HiddenMarkovModelState> stateMap,
                Map<HiddenMarkovModelState, Integer> inverseStateMap) {

            double[][] forwardMatrix = new double[sequence.length() + 1]
                                                 [stateMap.size()];

            // Set all required cells to unset sentinel:
            for (int rowIndex = 1; rowIndex < forwardMatrix.length; rowIndex++) {
                Arrays.fill(forwardMatrix[rowIndex], UNSET_PROBABILITY);
            }

            // BEGIN: Base case initialization.
            forwardMatrix[0][0] = 1.0;

            for (int columnIndex = 1; 
                     columnIndex < forwardMatrix[0].length;
                     columnIndex++) {

                forwardMatrix[0][columnIndex] = 0.0;
            }
            // END: Done with the base case initialization.

            // Recursively build the matrix:
            for (int h = 1; h < forwardMatrix[0].length - 1; h++) {
                forwardMatrix[sequence.length()][h] = 
                    computeForwardMatrixImpl(
                            sequence.length(),
                            h,
                            forwardMatrix, 
                            sequence, 
                            stateMap,
                            inverseStateMap);
            }

            return forwardMatrix;
        }
        
        private void loadStateMaps(
                Set<HiddenMarkovModelState> graph,
                Map<Integer, HiddenMarkovModelState> stateMap,
                Map<HiddenMarkovModelState, Integer> inverseStateMap) {

            // Initialize maps for start and end states:
            stateMap.put(0, startState);
            stateMap.put(graph.size() - 1, endState);

            inverseStateMap.put(startState, 0);
            inverseStateMap.put(endState, graph.size() - 1);

            int stateId = 1;

            // Assign indices to hidden states:
            for (HiddenMarkovModelState state : graph) {
                if (!state.equals(startState) && !state.equals(endState)) {
                    stateMap.put(stateId, state);
                    inverseStateMap.put(state, stateId);
                    stateId++;
                }
            }
        }

        /**
         * Computes the actual Viterbi matrix.
         * 
         * @param i             the {@code i} variable; the length of the sequence 
         *                      prefix.
         * @param h             the state index.
         * @param viterbiMatrix the actual Viterbi matrix under construction.
         * @param sequence      the symbol sequence.
         * @param stateMap      the map mapping state IDs to states.
         * @return 
         */
        private double computeViterbiMatrixImpl(
                int i,
                int h,
                double[][] viterbiMatrix,
                String sequence,
                Map<Integer, HiddenMarkovModelState> stateMap,
                Map<HiddenMarkovModelState, Integer> inverseStateMap) {

            if (viterbiMatrix[i][h] != UNSET_PROBABILITY) {
                return viterbiMatrix[i][h];
            }

            final int NUMBER_OF_STATES = stateMap.size();

            if (h >= NUMBER_OF_STATES - 1) {
                return UNSET_PROBABILITY;
            }

            if (h == 0) {
                return i == 0 ? 1.0 : -1.0;
            }

            char symbol = sequence.charAt(i - 1);

            HiddenMarkovModelState state = stateMap.get(h);
            double psih = state.getEmissions().get(symbol);

            Set<HiddenMarkovModelState> parentStates = state.getIncomingStates();

            double maximumProbability = Double.NEGATIVE_INFINITY;

            for (HiddenMarkovModelState parent : parentStates) {
                double v = 
                        computeViterbiMatrixImpl(
                                i - 1, 
                                inverseStateMap.get(parent),
                                viterbiMatrix, 
                                sequence,
                                stateMap, 
                                inverseStateMap);

                v *= parent.getFollowingStates().get(state);
                maximumProbability = Math.max(maximumProbability, v);
            }

            viterbiMatrix[i][h] = maximumProbability * psih;
            return viterbiMatrix[i][h];
        }

        /**
         * Computes the actual forward matrix.
         * 
         * @param i             the {@code i} variable; the length of the sequence 
         *                      prefix.
         * @param h             the state index.
         * @param forwardMatrix the actual Viterbi matrix under construction.
         * @param sequence      the symbol sequence.
         * @param stateMap      the map mapping state IDs to states.
         * 
         * @return the forward matrix.
         */
        private double computeForwardMatrixImpl(
                int i,
                int h,
                double[][] forwardMatrix,
                String sequence,
                Map<Integer, HiddenMarkovModelState> stateMap,
                Map<HiddenMarkovModelState, Integer> inverseStateMap) {

            if (forwardMatrix[i][h] != UNSET_PROBABILITY) {
                return forwardMatrix[i][h];
            }

            final int NUMBER_OF_STATES = stateMap.size();

            if (h >= NUMBER_OF_STATES - 1) {
                return UNSET_PROBABILITY;
            }

            if (h == 0) {
                return i == 0 ? 1.0 : 0.0;
            }

            char symbol = sequence.charAt(i - 1);

            HiddenMarkovModelState state = stateMap.get(h);
            double psih = state.getEmissions().get(symbol);

            Set<HiddenMarkovModelState> parentStates = state.getIncomingStates();

            double totalProbability = 0.0;

            for (HiddenMarkovModelState parent : parentStates) {
                double f =  
                        computeForwardMatrixImpl(
                                i - 1, 
                                inverseStateMap.get(parent),
                                forwardMatrix, 
                                sequence,
                                stateMap, 
                                inverseStateMap);

                f  *= parent.getFollowingStates().get(state);
                totalProbability += f ;
            }

            forwardMatrix[i][h] = totalProbability * psih;
            return forwardMatrix[i][h];
        }

        private HiddenMarkovModelStatePath 
            tracebackStateSequenceViterbi(
                    double[][] viterbiMatrix,
                    String sequence,
                    Map<Integer, HiddenMarkovModelState> stateMap,
                    Map<HiddenMarkovModelState, Integer> inverseStateMap) {

            int bottomMaximumIndex = computeBottomMaximumIndex(viterbiMatrix);

            HiddenMarkovModelState bottomState = stateMap.get(bottomMaximumIndex);

            List<HiddenMarkovModelState> stateList = 
                    new ArrayList<>(viterbiMatrix[0].length);

            stateList.add(endState);

            final int HIGHEST_I = viterbiMatrix.length - 1;

            tracebackStateSequenceImpl(viterbiMatrix,
                                       HIGHEST_I, 
                                       bottomState, 
                                       stateList, 
                                       inverseStateMap);

            stateList.add(startState);

            Collections.reverse(stateList);

            return new HiddenMarkovModelStatePath(stateList, sequence, this);
        }

        private double 
            tracebackStateSequenceForward(
                    double[][] forwardMatrix,
                    Map<HiddenMarkovModelState, Integer> inverseStateMap) {

            final int ROW_INDEX = forwardMatrix.length - 1;

            double probability = 0.0;

            Set<HiddenMarkovModelState> parents = endState.getIncomingStates();

            for (HiddenMarkovModelState parent : parents) {
                if (parent.equals(startState)) {
                    // Omit the start state:
                    continue;
                }

                int parentIndex = inverseStateMap.get(parent);
                double p = forwardMatrix[ROW_INDEX][parentIndex];
                p *= parent.getFollowingStates().get(endState);
                probability += p;
            }

            return probability;
        }

        private void tracebackStateSequenceImpl(
                double[][] viterbiMatrix,
                int i,
                HiddenMarkovModelState state,
                List<HiddenMarkovModelState> stateList,
                Map<HiddenMarkovModelState, Integer> inverseStateMap) {

            if (state.equals(startState)) {
                return;
            }

            stateList.add(state);

            HiddenMarkovModelState nextState = 
                    computeNextState(viterbiMatrix, 
                                     i,
                                     state,
                                     inverseStateMap);

            tracebackStateSequenceImpl(viterbiMatrix, 
                                       i - 1,
                                       nextState,
                                       stateList,
                                       inverseStateMap);
        }

        private HiddenMarkovModelState
             computeNextState(
                     double[][] viterbiMatrix, 
                     int i, 
                     HiddenMarkovModelState state,
                     Map<HiddenMarkovModelState, Integer> inverseStateMap) {

            Set<HiddenMarkovModelState> parents = state.getIncomingStates();
            HiddenMarkovModelState nextState = null;

            double maximumProbability = Double.NEGATIVE_INFINITY;

            for (HiddenMarkovModelState parent : parents) {
                int parentIndex = inverseStateMap.get(parent);

                double probability = parent.getFollowingStates().get(state);
                probability *= viterbiMatrix[i - 1][parentIndex];

                if (maximumProbability < probability) {
                    maximumProbability = probability;
                    nextState = parent;
                }
            }

            return nextState;
        }

        private int computeBottomMaximumIndex(double[][] viterbiMatrix) {
            int maximumIndex = -1;
            double maximumProbability = Double.NEGATIVE_INFINITY;
            final int ROW_INDEX = viterbiMatrix.length - 1;

            for (int i = 1; i < viterbiMatrix[0].length; i++) {
                double currentProbability = viterbiMatrix[ROW_INDEX][i];

                if (maximumProbability < currentProbability) {
                    maximumProbability = currentProbability;
                    maximumIndex = i;
                }
            }

            return maximumIndex;
        }

        private HiddenMarkovModelState 
            doStateTransition(HiddenMarkovModelState currentState) {

            double coin = random.nextDouble();

            for (Map.Entry<HiddenMarkovModelState, Double> e 
                    : currentState.getFollowingStates().entrySet()) {

                if (coin >= e.getValue()) {
                    coin -= e.getValue();
                } else {
                    return e.getKey();
                }
            }

            throw new IllegalStateException("Should not get here.");
        }

        private char doEmit(HiddenMarkovModelState currentState) {
            double coin = random.nextDouble();

            for (Map.Entry<Character, Double> e 
                    : currentState.getEmissions().entrySet()) {

                if (coin >= e.getValue()) {
                    coin -= e.getValue();
                } else {
                    return e.getKey();
                }
            }

            throw new IllegalStateException("Should not get here.");
        }

        private Set<HiddenMarkovModelState> computeAllStates() {
            Deque<HiddenMarkovModelState> queue = new ArrayDeque<>();
            Set<HiddenMarkovModelState> visited = new HashSet<>();

            queue.addLast(startState);
            visited.add(startState);

            while (!queue.isEmpty()) {
                HiddenMarkovModelState currentState = queue.removeFirst();

                for (HiddenMarkovModelState followerState 
                        : currentState.getFollowingStates().keySet()) {

                    if (!visited.contains(followerState)) {
                        visited.add(followerState);
                        queue.addLast(followerState);
                    }
                }
            }

            return visited;
        }

        private void depthFirstSearchImpl(
                List<List<HiddenMarkovModelState>> statePaths,
                List<HiddenMarkovModelState> currentPath,
                int expectedStatePathSize,
                HiddenMarkovModelState currentState) {

            if (currentPath.size() == expectedStatePathSize) {
                // End recursion. If the current state equals the end state, we have
                // a path:
                if (currentState.equals(endState)) {
                    statePaths.add(new ArrayList<>(currentPath));
                }

                return;
            }

            // For each child, do...
            for (HiddenMarkovModelState followerState 
                    : currentState.getFollowingStates().keySet()) {

                // Do state:
                currentPath.add(followerState);

                // Recur deeper:
                depthFirstSearchImpl(statePaths,
                                     currentPath,
                                     expectedStatePathSize, 
                                     followerState);

                // Undo state:
                currentPath.remove(currentPath.size() - 1);
            }
        }
    }
