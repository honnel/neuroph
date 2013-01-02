package edu.kit.pmk.neuroph.parallel.networkclones.interpolation;

import org.neuroph.core.Weight;

public class MinimumNeuralNetInterpolator extends NeuralNetInterpolator {

	@Override
	public WeightAdjustment getWeightAdjustmentForIndex(
			Weight[][] neuronWeights, int index) {
		WeightAdjustment min = new WeightAdjustment();
		min.adjustedWeight = Double.MAX_VALUE;
		min.adjustedWeightChange = Double.MAX_VALUE;
		for (int net = 0; net < neuronWeights.length; net++) {
			min.adjustedWeight = Math.min(min.adjustedWeight,
					neuronWeights[net][index].value);
			min.adjustedWeightChange = Math.min(min.adjustedWeightChange,
					neuronWeights[net][index].weightChange);
		}
		return min;
	}

}
