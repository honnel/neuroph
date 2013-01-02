package edu.kit.pmk.neuroph.parallel.networkclones.interpolation;

import org.neuroph.core.Weight;

public class ArithmeticMeanNeuralNetInterpolator extends NeuralNetInterpolator {

	public ArithmeticMeanNeuralNetInterpolator() {
		super();
	}

	@Override
	public WeightAdjustment getWeightAdjustmentForIndex(
			Weight[][] neuronWeights, int index) {
		WeightAdjustment avg = new WeightAdjustment();
		avg.adjustedWeight = 0;
		avg.adjustedWeightChange = 0;
		for (int net = 0; net < neuronWeights.length; net++) {
			avg.adjustedWeight += neuronWeights[net][index].value;
			avg.adjustedWeightChange += neuronWeights[net][index].weightChange;
		}
		avg.adjustedWeight /= neuronWeights.length;
		avg.adjustedWeightChange /= neuronWeights.length;
		return avg;
	}

}
