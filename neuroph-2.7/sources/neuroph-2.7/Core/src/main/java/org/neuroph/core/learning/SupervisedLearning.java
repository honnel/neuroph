/**
 * Copyright 2010 Neuroph Project http://neuroph.sourceforge.net
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.neuroph.core.learning;

import java.io.Serializable;
import java.util.Iterator;
import org.neuroph.core.Connection;
import org.neuroph.core.Layer;
import org.neuroph.core.Neuron;
import org.neuroph.core.Weight;

// TODO:  random pattern order
/**
 * Base class for all supervised learning algorithms.
 * It extends IterativeLearning, and provides general supervised learning principles.
 * 
 * @author Zoran Sevarac <sevarac@gmail.com>
 */
abstract public class SupervisedLearning extends IterativeLearning implements
        Serializable {

    /**
     * The class fingerprint that is set to indicate serialization 
     * compatibility with a previous version of the class
     */
    private static final long serialVersionUID = 3L;
    /**
     * Total network error
     */
    protected transient double totalNetworkError;
    /**
     * Total squared sum of all pattern errors
     */
    protected transient double totalSquaredErrorSum;
    /**
     * Total network error in previous epoch
     */
    protected transient double previousEpochError;
    /**
     * Max allowed network error (condition to stop learning)
     */
    protected double maxError = 0.01d;
    /**
     * Stopping condition: training stops if total network error change is smaller than minErrorChange
     * for minErrorChangeIterationsLimit number of iterations
     */
    private double minErrorChange = Double.POSITIVE_INFINITY;
    /**
     * Stopping condition: training stops if total network error change is smaller than minErrorChange
     * for minErrorChangeStopIterations number of iterations
     */
    private int minErrorChangeIterationsLimit = Integer.MAX_VALUE;
    /**
     * Count iterations where error change is smaller then minErrorChange
     */
    private transient int minErrorChangeIterationsCount;
    /**
     * Setting to determine if learning (weights update) is in batch mode
     * False by default.
     */
    private boolean batchMode = false;
    
    /**
     * Stores network output error vector
     */
   // protected double[] outputError;    
    
    private int trainingSetSize;

    /**
     * Creates new supervised learning rule
     */
    public SupervisedLearning() {
        super();
    }

    /**
     * Trains network for the specified training set and number of iterations
     * @param trainingSet training set to learn
     * @param maxError maximum number of iterations to learn
     *
     */
    public void learn(DataSet trainingSet, double maxError) {
        this.maxError = maxError;
        this.learn(trainingSet);
    }

    /**
     * Trains network for the specified training set and number of iterations
     * @param trainingSet training set to learn
     * @param maxIterations maximum number of learning iterations
     *
     */
    public void learn(DataSet trainingSet, double maxError, int maxIterations) {
        this.maxError = maxError;
        this.setMaxIterations(maxIterations);
        this.learn(trainingSet);
    }

    @Override
    protected void onStart() {
        super.onStart(); // reset iteration counter
        this.minErrorChangeIterationsCount = 0;
        this.totalNetworkError = 0d;
        this.previousEpochError = 0d;
        //this.outputError = new double[neuralNetwork.getOutputsCount()]; // initialize output error buffer        
    }
    
    @Override
    protected void beforeEpoch() {
        this.previousEpochError = this.totalNetworkError;
        this.totalNetworkError = 0d;
        this.totalSquaredErrorSum = 0d;      
        this.trainingSetSize = getTrainingSet().size();
    }
    
    @Override
    protected void afterEpoch() {
        // if learning is performed in batch mode, apply accumulated weight changes from this epoch
        if (this.batchMode == true) {
            doBatchWeightsUpdate();
        }        
    }

    /**
     * This method implements basic logic for one learning epoch for the
     * supervised learning algorithms. Epoch is the one pass through the
     * training set. This method  iterates through the training set
     * and trains network for each element. It also sets flag if conditions 
     * to stop learning has been reached: network error below some allowed
     * value, or maximum iteration count 
     * 
     * @param trainingSet
     *            training set for training network
     */
    @Override
    public void doLearningEpoch(DataSet trainingSet) {
        
        // feed network with all elements from training set
        Iterator<DataSetRow> iterator = trainingSet.iterator();
        while (iterator.hasNext() && !isStopped()) {
            DataSetRow dataSetRow = iterator.next();
            // learn current input/output pattern defined by SupervisedTrainingElement
            this.learnPattern(dataSetRow); 
        }

        // calculate total network error as MSE. Use MSE so network does not grow with bigger training sets
        this.totalNetworkError = this.totalSquaredErrorSum / this.trainingSetSize;

        // moved stopping condition to separate method hasReachedStopCondition() so it can be overriden / customized in subclasses
        if (hasReachedStopCondition()) {
            stopLearning();
        }
    }

    /**
     * Trains network with the input and desired output pattern from the specified training element
     * 
     * @param trainingElement
     *            supervised training element which contains input and desired
     *            output
     */
    protected void learnPattern(DataSetRow trainingElement) {
        double[] input = trainingElement.getInput();
        this.neuralNetwork.setInput(input);
        this.neuralNetwork.calculate();
        double[] output = this.neuralNetwork.getOutput();
        double[] desiredOutput = trainingElement.getDesiredOutput();
        double[] outputError = this.calculateOutputError(desiredOutput, output);
        this.addToSquaredErrorSum(outputError);
        this.updateNetworkWeights(outputError);
    }
    
    /**
     * This method updates network weights in batch mode - use accumulated weights change stored in Weight.deltaWeight
     * It is executed after each learning epoch, only if learning is done in batch mode.
     * @see SupervisedLearning#doLearningEpoch(org.neuroph.core.learning.TrainingSet)
     */
    protected void doBatchWeightsUpdate() {
        // iterate layers from output to input
        Layer[] layers = neuralNetwork.getLayers();
        for (int i = neuralNetwork.getLayersCount() - 1; i > 0; i--) {
            // iterate neurons at each layer
            for (Neuron neuron : layers[i].getNeurons()) {
                // iterate connections/weights for each neuron
                for (Connection connection : neuron.getInputConnections()) {
                    // for each connection weight apply accumulated weight change
                    Weight weight = connection.getWeight();
                    weight.value += weight.weightChange; // apply delta weight which is the sum of delta weights in batch mode
                    weight.weightChange = 0; // reset deltaWeight
                }
            }
        }
    }    
        

    /**
     * Returns true if stop condition has been reached, false otherwise.
     * Override this method in derived classes to implement custom stop criteria.
     *
     * @return true if stop condition is reached, false otherwise
     */
    protected boolean hasReachedStopCondition() {
        // da li ovd etreba staviti da proverava i da li se koristi ovaj uslov??? ili staviti da uslov bude automatski samo s ajaako malom vrednoscu za errorChange Doule.minvalue
        return (this.totalNetworkError < this.maxError) || this.errorChangeStalled();
    }

    /**
     * Returns true if absolute error change is sufficently small (<=minErrorChange) for minErrorChangeStopIterations number of iterations
     * @return true if absolute error change is stalled (error is sufficently small for some number of iterations)
     */
    protected boolean errorChangeStalled() {
        double absErrorChange = Math.abs(previousEpochError - totalNetworkError);

        if (absErrorChange <= this.minErrorChange) {
            this.minErrorChangeIterationsCount++;

            if (this.minErrorChangeIterationsCount >= this.minErrorChangeIterationsLimit) {
                return true;
            }
        } else {
            this.minErrorChangeIterationsCount = 0;
        }

        return false;
    }

    /**
     * Calculates the network error for the current input pattern - diference between
     * desired and actual output
     * 
     * @param output
     *            actual network output
     * @param desiredOutput
     *            desired network output
     */
    protected double[] calculateOutputError(double[] desiredOutput, double[] output) {
        double[] outputError = new double[desiredOutput.length];
        
        for (int i = 0; i < output.length; i++) {
            outputError[i] = desiredOutput[i] - output[i];
        }
        
        return outputError;
    }
    
    /**
     * Returns true if learning is performed in batch mode, false otherwise
     * @return true if learning is performed in batch mode, false otherwise
     */
    public boolean isInBatchMode() {
        return batchMode;
    }

    /**
     * Sets batch mode on/off (true/false)
     * @param batchMode batch mode setting
     */
    public void setBatchMode(boolean batchMode) {
        this.batchMode = batchMode;
    }    

    /**
     * Sets allowed network error, which indicates when to stopLearning training
     * 
     * @param maxError
     *            network error
     */
    public void setMaxError(double maxError) {
        this.maxError = maxError;
    }

    /**
     * Returns learning error tolerance - the value of total network error to stop learning.
     *
     * @return learning error tolerance
     */
    public double getMaxError() {
        return maxError;
    }

    /**
     * Returns total network error in current learning epoch
     * 
     * @return total network error in current learning epoch
     */
    public synchronized double getTotalNetworkError() {
        return totalNetworkError;
    }

    /**
     * Returns total network error in previous learning epoch
     *
     * @return total network error in previous learning epoch
     */
    public double getPreviousEpochError() {
        return previousEpochError;
    }

    /**
     * Returns min error change stopping criteria
     *
     * @return min error change stopping criteria
     */
    public double getMinErrorChange() {
        return minErrorChange;
    }

    /**
     * Sets min error change stopping criteria
     *
     * @param minErrorChange value for min error change stopping criteria
     */
    public void setMinErrorChange(double minErrorChange) {
        this.minErrorChange = minErrorChange;
    }

    /**
     * Returns number of iterations for min error change stopping criteria
     *
     * @return number of iterations for min error change stopping criteria
     */
    public int getMinErrorChangeIterationsLimit() {
        return minErrorChangeIterationsLimit;
    }

    /**
     * Sets number of iterations for min error change stopping criteria
     * @param minErrorChangeIterationsLimit number of iterations for min error change stopping criteria
     */
    public void setMinErrorChangeIterationsLimit(int minErrorChangeIterationsLimit) {
        this.minErrorChangeIterationsLimit = minErrorChangeIterationsLimit;
    }

    /**
     * Returns number of iterations count for for min error change stopping criteria
     *
     * @return number of iterations count for for min error change stopping criteria
     */
    public int getMinErrorChangeIterationsCount() {
        return minErrorChangeIterationsCount;
    }

    /**
     * Calculates and updates sum of squared errors for single pattern, and updates total sum of squared pattern errors
     *
     * @param outputError output error vector
     */
    // see: http://www.vni.com/products/imsl/documentation/CNL06/stat/NetHelp/default.htm?turl=multilayerfeedforwardneuralnetworks.htm
    protected void addToSquaredErrorSum(double[] outputError) {
        double outputErrorSqrSum = 0;
        for (double error : outputError) {
            outputErrorSqrSum += (error * error) * 0.5; // a;so multiply with 1/trainingSetSize  1/2n * (...)
        }

        this.totalSquaredErrorSum += outputErrorSqrSum;
    }

    /**
     * This method should implement the weights update procedure for the whole network
     * for the given output error vector.
     * 
     * @param outputError
     *            output error vector for some network input (aka. patternError, network error) 
     *            usually the difference between desired and actual output
     *
     * @see SupervisedLearning#calculateOutputError(double[], double[])  calculateOutputError 
     * @see SupervisedLearning#addToSquaredErrorSum(double[])
     */
    abstract protected void updateNetworkWeights(double[] outputError);
}
