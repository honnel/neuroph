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

package edu.kit.pmk.neuroph.parallel.IrisClassificationSample;

import java.util.Arrays;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.learning.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BatchParallelMomentumBackpropagation;
import org.neuroph.core.learning.DataSet;

/**
 * This sample shows how to train MultiLayerPerceptron neural network for iris classification problem using Neuroph
 * For more details about training process, error, iterations use NeurophStudio which provides rich environment  for
 * training and inspecting neural networks
 * @author Zoran Sevarac <sevarac@gmail.com>
 */
public class IrisClassificationSampleBatch {  
    
    /**
     *  Runs this sample
     */
    public static void main(String[] args) {    
        // get the path to file with data
        String inputFileName = org.neuroph.samples.IrisClassificationSample.class.getResource("data/iris_data_normalised.txt").getFile();
        
        // create MultiLayerPerceptron neural network
        MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(4, 16, 3);
        neuralNet.setLearningRule(new BatchParallelMomentumBackpropagation(15));
        // create training set from file
        DataSet irisDataSet = DataSet.createFromFile(inputFileName, 4, 3, ",");
        // train the network with training set
        neuralNet.learn(irisDataSet);         
        
        System.out.println("Done training.");
        System.out.println("Testing network...");
        
        testNeuralNetwork(neuralNet, irisDataSet);
    }
    
    /**
     * Prints network output for the each element from the specified training set.
     * @param neuralNet neural network
     * @param testSet test data set
     */
    public static void testNeuralNetwork(NeuralNetwork neuralNet, DataSet testSet) {

        for(DataSetRow testSetRow : testSet.getRows()) {
            neuralNet.setInput(testSetRow.getInput());
            neuralNet.calculate();
            double[] networkOutput = neuralNet.getOutput();

            System.out.print("Input: " + Arrays.toString( testSetRow.getInput() ) );
            System.out.print(" Output: " + Arrays.toString( networkOutput) );
        }
    }
    
}
