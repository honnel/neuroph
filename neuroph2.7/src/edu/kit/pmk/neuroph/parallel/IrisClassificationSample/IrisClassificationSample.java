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
import org.neuroph.core.Neuron;
import org.neuroph.core.learning.DataSet;
import org.neuroph.core.learning.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.LMS;

/**
 * This sample shows how to train MultiLayerPerceptron neural network for iris
 * classification problem using Neuroph For more details about training process,
 * error, iterations use NeurophStudio which provides rich environment for
 * training and inspecting neural networks
 * 
 * @author Zoran Sevarac <sevarac@gmail.com>
 */
public class IrisClassificationSample {

	/**
	 * Runs this sample
	 * 
	 * @throws InterruptedException
	 */
	public static void main(String[] args) throws InterruptedException {
		IrisClassificationSample irisLearner = new IrisClassificationSample();

		// get the path to file with data
		String inputFileName = org.neuroph.samples.IrisClassificationSample.class
				.getResource("data/iris_data_normalised.txt").getFile();
		// create training set from file
		DataSet irisDataSet = DataSet.createFromFile(inputFileName, 4, 3, ",");

		// create MultiLayerPerceptron neural network
		MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(4, 16, 3);
		((LMS) neuralNet.getLearningRule()).setMaxIterations(100);

		irisLearner.learnParallel(2, 10, neuralNet, irisDataSet);

		System.err.println("Done training");
		System.out.println("Testing network...");
		testNeuralNetwork(neuralNet, irisDataSet);
	}

	public void learnSequential(NeuralNetwork neuralNet, DataSet dataSet) {
		// train the network with training set
		neuralNet.learn(dataSet);
	}

	/**
	 * Trains the network with <tt>numThreads</tt> threads concurrently. The
	 * <tt>dataSet</tt> is split into sets of equal size. Each thread trains a
	 * clone of <tt>neuralNet</tt> with its subset of <tt>dataSet</tt>. After
	 * <tt>syncFrequency</tt> training elements the threads synchronize and
	 * interpolate their neural network clones. Interpolation means that each
	 * weight and weightChange property is assigned the average weight and
	 * average weightChange from the neural network clones.
	 * 
	 * @param numThreads
	 * @param syncFrequency
	 *            determines how many training elements are processed before an
	 *            interpolation occurs
	 * @param neuralNet
	 *            the neural network to be trained
	 * @param dataSet
	 * @throws InterruptedException
	 */
	public void learnParallel(int numThreads, int syncFrequency,
			NeuralNetwork neuralNet, DataSet dataSet)
			throws InterruptedException {
		NeuralNetwork[] nets = createNeuralNetworkClones(numThreads, neuralNet);

		DataSet[] dataSets = splitDataSet(numThreads, dataSet);

		final long t0 = System.currentTimeMillis();
		final NeuralNetInterpolator interpolator = new NeuralNetInterpolator(
				nets);

		final CyclicBarrier barrier = new CyclicBarrier(nets.length,
				new Runnable() {

					@Override
					public void run() {
						System.out.println("Interpolate!");
						interpolator.interpolateWeights();
						System.out.println("Current time is "
								+ (System.currentTimeMillis() - t0) + "ms.");
					}
				});

		Thread[] workers = initializeAndStartWorkers(syncFrequency, dataSets,
				nets, barrier);
		waitForWorkersCompletion(workers);
	}

	private DataSet[] splitDataSet(int numSubsets, DataSet dataSet) {
		DataSet[] dataSets = new DataSet[numSubsets];
		for (int i = 0; i < dataSets.length; i++) {
			dataSets[i] = new DataSet(dataSet.getInputSize(),
					dataSet.getOutputSize());
		}
		int rowIndex = 0;
		for (DataSetRow row : dataSet.getRows()) {
			dataSets[rowIndex % dataSets.length].addRow(row);
			rowIndex++;
		}
		return dataSets;
	}

	private NeuralNetwork[] createNeuralNetworkClones(int numClones,
			NeuralNetwork neuralNet) {
		NeuralNetwork[] nets = new NeuralNetwork[numClones];
		nets[0] = neuralNet;
		for (int i = 1; i < nets.length; i++) {
			try {
				nets[i] = (NeuralNetwork) DeepCopy.createDeepCopy(neuralNet);
			} catch (ClassNotFoundException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		return nets;
	}

	private void waitForWorkersCompletion(Thread[] workers)
			throws InterruptedException {
		for (int i = 0; i < workers.length; i++) {
			workers[i].join();
		}
	}

	private Thread[] initializeAndStartWorkers(int syncFrequency,
			DataSet[] dataSets, NeuralNetwork[] nets,
			final CyclicBarrier barrier) {
		Thread[] workers = new Thread[nets.length];
		for (int i = 0; i < workers.length; i++) {
			workers[i] = new Thread(new ShadowNetWorker(barrier, nets[i],
					dataSets[i], syncFrequency));
		}
		for (int i = 0; i < workers.length; i++) {
			workers[i].start();
		}
		return workers;
	}

	/**
	 * Prints network output for the each element from the specified training
	 * set.
	 * 
	 * @param neuralNet
	 *            neural network
	 * @param testSet
	 *            test data set
	 */
	public static void testNeuralNetwork(NeuralNetwork neuralNet,
			DataSet testSet) {

		for (DataSetRow testSetRow : testSet.getRows()) {
			neuralNet.setInput(testSetRow.getInput());
			neuralNet.calculate();
			double[] networkOutput = neuralNet.getOutput();

			System.out
					.print("Input: " + Arrays.toString(testSetRow.getInput()));
			System.out.print(" Output: " + Arrays.toString(networkOutput));
			StringBuilder error = new StringBuilder(" Error: [");
			for (Neuron neuron : neuralNet.getOutputNeurons()) {
				error.append(neuron.getError());
				error.append(", ");
			}
			error.delete(error.length() - 2, error.length());
			error.append(']');
			System.out.println(error);
		}
	}

}
