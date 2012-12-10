package edu.kit.pmk.neuroph.eval.generality;

import java.io.IOException;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.learning.DataSet;
import org.neuroph.core.learning.DataSetRow;

//import edu.kit.pmk.neuroph.parallel.networkclones.DeepCopy;
import edu.kit.pmk.neuroph.parallel.ILearner;
import edu.kit.pmk.neuroph.parallel.networkclones.FastDeepCopy;

public class GeneralityScore {

	public static Score trainAndCalculateError(ILearner learner,
			DataSet dataSet, double trainingSetRatio, int runs) {
		double error = 0;
		long t0 = System.currentTimeMillis();

		for (int i = 0; i < runs; i++) {
			TestAndTrainingSet tats = TestAndTrainingSet.splitSetAndPermute(
					dataSet, trainingSetRatio);
//			try {
//				//netClone = (NeuralNetwork) DeepCopy.createDeepCopy(neuralNet);
//				netClone = (NeuralNetwork) FastDeepCopy.createDeepCopy(neuralNet);
//			} catch (ClassNotFoundException e1) {
//				e1.printStackTrace();
//			} catch (IOException e1) {
//				e1.printStackTrace();
//			}
			learner.resetToUnlearnedState();
			learner.learn(tats.getTrainingSet());
//			ClonebasedConcurrentLearner ccl = new ClonebasedConcurrentLearner();
//			try {
//				ccl.learnParallel(2, 8, netClone, tats.getTrainingSet());
//			} catch (InterruptedException e) {
//				e.printStackTrace();
//			}
			error += calculateError(learner, tats.getTestSet());
		}
		long time = System.currentTimeMillis() - t0;
		error =  error / runs;
		return new Score(error, time);
	}

	private static double calculateError(ILearner learner,
			DataSet testSet) {
		double error = 0;
		NeuralNetwork neuralNet = learner.getNeuralNetwork();
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
