package edu.kit.pmk.neuroph.parallel.networkclones.interpolation;

import org.neuroph.core.Weight;
import org.neuroph.nnet.learning.LMS;

public class GeneticNeuralNetInterpolator extends NeuralNetInterpolator {

	public GeneticNeuralNetInterpolator() {
		super();
	}

	@Override
	public WeightAdjustment getWeightAdjustmentForIndex(
			Weight[][] neuronWeights, int index) {
		int bestNetIndex = getBestNetwork();
		WeightAdjustment best = new WeightAdjustment();
		best.adjustedWeight = neuronWeights[bestNetIndex][index].value;
		best.adjustedWeightChange = neuronWeights[bestNetIndex][index].weightChange;
		return best;
	}

	private int getBestNetwork() {
		int bestNeuralNetIndex = -1;
		double besterror = Double.MAX_VALUE;
		for (int i = 0; i < workers.length; i++) {
			if (((LMS) workers[i].getNeuralNetwork().getLearningRule())
					.getTotalNetworkError() <= besterror) {
				bestNeuralNetIndex = i;
			}
		}
		return bestNeuralNetIndex;
	}

}
