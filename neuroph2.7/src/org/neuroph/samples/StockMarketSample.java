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

package org.neuroph.samples;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.learning.DataSet;
import org.neuroph.core.learning.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.LMS;
import org.neuroph.util.TransferFunctionType;

/**
 * This sample shows how to train MultiLayerPerceptron neural network for iris
 * classification problem using Neuroph For more details about training process,
 * error, iterations use NeurophStudio which provides rich environment for
 * training and inspecting neural networks
 * 
 * @author Zoran Sevarac <sevarac@gmail.com>
 */
public class StockMarketSample {
	public static final double maxValueUp = 300.0;
	public static final double maxValueDown = 300.0;

	/**
	 * Runs this sample
	 */
	public static void main(String[] args) {
		// get the path to file with data
		String inputFileName = StockMarketSample.class.getResource(
				"data/stockmarket.txt").getFile();
		Scanner sc;
		try {
			sc = new Scanner(new File(inputFileName));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			return;
		}
		int useDiffsInPast = 3;
		int diffsToPredict = 1;

		double currentNumber = -1;
		double nextNumber = -1;
		List<Double> diffs = new ArrayList<Double>();

		if (sc.hasNextDouble()) {
			currentNumber = sc.nextDouble();
		}
		while (sc.hasNextDouble()) {
			nextNumber = sc.nextDouble();
			diffs.add(nextNumber - currentNumber);
			currentNumber = nextNumber;
		}

		DataSet trainingData = new DataSet(useDiffsInPast, diffsToPredict);

		for (int i = 0; i < diffs.size() - useDiffsInPast - diffsToPredict; i++) {
			double[] input = new double[useDiffsInPast];
			double[] output = new double[diffsToPredict];
			double min = diffs.get(i), max = min;

			for (int j = 0; j < useDiffsInPast; j++) {
				input[j] = diffs.get(i + j);
				input[j] = (input[j] + maxValueDown)
						/ (maxValueUp + maxValueDown);
				// min = input[j] < min ? input[j] : min;
				// max = input[j] > max ? input[j] : max;
				System.out.print(input[j] + " ");
			}

			System.out.print(" : ");
			for (int j = 0; j < output.length; j++) {
				output[j] = diffs.get(i + j + useDiffsInPast);
				output[j] = (output[j] + maxValueDown)
						/ (maxValueUp + maxValueDown);
				// min = output[j] < min ? output[j] : min;
				// max = output[j] > max ? output[j] : max;
				System.out.print(output[j] + " ");
			}
			max *= 1.2;
			min *= 0.8;
			System.out.println();

			// for (int j = 0; j < input.length; j++) {
			// input[j] = (input[j] - min) / (max - min) * 0.8 + 0.1;
			// input[j] = (input[j] - min) / max;
			// System.out.print(input[j] + " ");
			// }
			// System.out.print(" : ");
			// for (int j = 0; j < output.length; j++) {
			// output[j] = (output[j] - min) / (max - min) * 0.8 + 0.1;
			// output[j] = (output[j] - min) / max;
			// System.out.print(output[j] + " ");
			// }
			// System.out.println();

			DataSetRow row = new DataSetRow(input, output);
			trainingData.addRow(row);
		}
//		System.exit(0);

		// create MultiLayerPerceptron neural network
		// MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(
		// useDaysInPast, 2 * useDaysInPast - 1, daysToPredict);

		NeuralNetwork neuralNet = new MultiLayerPerceptron(
				TransferFunctionType.GAUSSIAN, useDiffsInPast,
				2 * useDiffsInPast + 1, diffsToPredict);
		((LMS) neuralNet.getLearningRule()).setMaxError(0.01);// 0-1
		((LMS) neuralNet.getLearningRule()).setLearningRate(0.5);// 0-1
		((LMS) neuralNet.getLearningRule()).setMaxIterations(100000);
		// create training set from file
		// train the network with training set
		neuralNet.learn(trainingData);

		System.out.println("Done training.");
		System.out.println("Testing network...");

		testNeuralNetwork(neuralNet, trainingData);
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
			System.out.println(" Output: " + Arrays.toString(networkOutput));
		}
	}

}
