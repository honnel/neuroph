package edu.kit.pmk.neuroph.eval;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.learning.DataSet;
import org.neuroph.core.learning.DataSetRow;

import edu.kit.pmk.neuroph.parallel.ILearner;

public class ScoreCalculator {

	public static Score trainAndCalculateOnPermutedSet(ILearner learner,
			DataSet dataSet, double trainingSetRatio, int runs) {
		double error = 0;
		long time = 0;

		for (int i = 0; i < runs; i++) {
			TestAndTrainingSet tats = TestAndTrainingSet.splitSetAndPermute(
					dataSet, trainingSetRatio);
			learner.resetToUnlearnedState();
			long t0 = System.currentTimeMillis();
			learner.learn(tats.getTrainingSet());
			time += System.currentTimeMillis() - t0;
			error += calculateError(learner, tats.getTestSet());
		}
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
