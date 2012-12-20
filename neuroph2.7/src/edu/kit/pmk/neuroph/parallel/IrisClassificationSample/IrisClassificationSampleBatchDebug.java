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

import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.CyclicBarrier;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.learning.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.nnet.learning.BatchParallelMomentumBackpropagation;
import org.neuroph.core.learning.DataSet;

import edu.kit.pmk.neuroph.parallel.networkclones.FastDeepCopy;

/**
 * This sample shows how to train MultiLayerPerceptron neural network for iris classification problem using Neuroph
 * For more details about training process, error, iterations use NeurophStudio which provides rich environment  for
 * training and inspecting neural networks
 * @author Zoran Sevarac <sevarac@gmail.com>
 */
public class IrisClassificationSampleBatchDebug {  
	public static NeuralNetwork firstNet;
	public static NeuralNetwork secondNet;
    public static final CyclicBarrier globalDebuggingBarrier = new CyclicBarrier(2, new Runnable() {
		@Override
		public void run() {
//			for(int i=0; i<firstNet.getLayersCount(); i++) {
//				for(int j=0; j<firstNet.getLayerAt(i).getNeuronsCount(); j++) {
//					for(int k=0; k<firstNet.getLayerAt(i).getNeuronAt(j).getInputConnections().length;k++) {
//						System.out.println(i + " " + j + " " + k);
//						System.out.print("vf: " + firstNet.getLayerAt(i).getNeuronAt(j).getWeights()[k].value);
//						System.out.println(" wf: " + firstNet.getLayerAt(i).getNeuronAt(j).getWeights()[k].weightChange);
//						System.out.print("vs: " + secondNet.getLayerAt(i).getNeuronAt(j).getWeights()[k].value);
//						System.out.println(" ws: " + secondNet.getLayerAt(i).getNeuronAt(j).getWeights()[k].weightChange);
//					}
//				}
//			}
			System.out.print("vf: " + firstNet.getLayerAt(1).getNeuronAt(2).getWeights()[2].value);
			System.out.println(" wf: " + firstNet.getLayerAt(1).getNeuronAt(2).getWeights()[2].weightChange);
			System.out.print("vs: " + secondNet.getLayerAt(1).getNeuronAt(2).getWeights()[2].value);
			System.out.println(" ws: " + secondNet.getLayerAt(1).getNeuronAt(2).getWeights()[2].weightChange);
			System.out.println("round over");
		}
	});
    /**
     *  Runs this sample
     */
    public static void main(String[] args) {    
        // get the path to file with data
        String inputFileName = org.neuroph.samples.IrisClassificationSample.class.getResource("data/iris_data_normalised.txt").getFile();
        
        MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(4, 16, 3);
		firstNet = neuralNet;
        
        
        // create training set from file
        final DataSet irisDataSet = DataSet.createFromFile(inputFileName, 4, 3, ",");
        
        try {
			final MultiLayerPerceptron debugClone = (MultiLayerPerceptron) FastDeepCopy.createDeepCopy(neuralNet);
			((BackPropagation) debugClone.getLearningRule()).setBatchMode(true);
			secondNet = debugClone;
			Thread debugThread = new Thread() {
	        	public void run() {
	        		debugClone.learn(irisDataSet);
	        	}
	        };
	        debugThread.start();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
        
        neuralNet.setLearningRule(new BatchParallelMomentumBackpropagation(8));
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
            System.out.println(" Output: " + Arrays.toString( networkOutput) );
        }
    }
    
}
