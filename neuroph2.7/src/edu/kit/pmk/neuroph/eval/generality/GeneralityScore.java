package edu.kit.pmk.neuroph.eval.generality;

import java.io.IOException;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.learning.DataSet;
import org.neuroph.core.learning.DataSetRow;

//import edu.kit.pmk.neuroph.parallel.networkclones.DeepCopy;
import edu.kit.pmk.neuroph.parallel.networkclones.FastDeepCopy;

public class GeneralityScore {

	public static double trainAndCalculateError(NeuralNetwork neuralNet,
			DataSet dataSet, double trainingSetRatio, int runs) {
		double error = 0;

		for (int i = 0; i < runs; i++) {
			TestAndTrainingSet tats = TestAndTrainingSet.splitSetAndPermute(
					dataSet, trainingSetRatio);
			NeuralNetwork netClone = null;
			try {
				//netClone = (NeuralNetwork) DeepCopy.createDeepCopy(neuralNet);
				netClone = (NeuralNetwork) FastDeepCopy.createDeepCopy(neuralNet);
			} catch (ClassNotFoundException e1) {
				e1.printStackTrace();
			} catch (IOException e1) {
				e1.printStackTrace();
			}
			netClone.learn(tats.getTrainingSet());
//			ClonebasedConcurrentLearner ccl = new ClonebasedConcurrentLearner();
//			try {
//				ccl.learnParallel(2, 8, netClone, tats.getTrainingSet());
//			} catch (InterruptedException e) {
//				e.printStackTrace();
//			}
			error += calculateError(neuralNet, tats.getTestSet());
		}
		return error / runs;
	}

	private static double calculateError(NeuralNetwork neuralNet,
			DataSet testSet) {
		double error = 0;
		for (DataSetRow testSetRow : testSet.getRows()) {
			neuralNet.setInput(testSetRow.getInput());
			neuralNet.calculate();
			for (Neuron neuron : neuralNet.getOutputNeurons()) {
				error += neuron.getError();
			}
		}
		return error / testSet.size();
	}
}
