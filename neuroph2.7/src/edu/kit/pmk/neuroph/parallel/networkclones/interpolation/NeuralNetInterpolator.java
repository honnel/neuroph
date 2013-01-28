package edu.kit.pmk.neuroph.parallel.networkclones.interpolation;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Weight;

import edu.kit.pmk.neuroph.parallel.networkclones.CloneNetWorker;

public abstract class NeuralNetInterpolator {

	protected CloneNetWorker[] workers;

	protected NeuralNetInterpolator() {
	}

	public static NeuralNetInterpolator createNeuralNetInterpolator(
			NeuralNetInterpolatorType type) {
		switch (type) {
		case ArithmeticMean:
			return new ArithmeticMeanNeuralNetInterpolator();
		case Minimum:
			return new MinimumNeuralNetInterpolator();
		case Maximum:
			return new MaximumNeuralNetInterpolator();
		case GeometricMean:
			return new GeometricMeanNeuralNetInterpolator();
		case Genetic:
			return new GeneticNeuralNetInterpolator();
		default:
			return new ArithmeticMeanNeuralNetInterpolator();
		}
	}

	public void setWorkers(CloneNetWorker[] workers) {
		this.workers = workers;
	}

	public void interpolateWeights() {
		NeuralNetwork net = workers[0].getNeuralNetwork();
		for (int layer = 0; layer < net.getLayersCount(); layer++) {
			for (int neuron = 0; neuron < net.getLayerAt(layer)
					.getNeuronsCount(); neuron++) {
				interpolateWeightAndWeightChangeForNeuron(layer, neuron);
			}
		}
	}

	private void interpolateWeightAndWeightChangeForNeuron(int layerIndex,
			int neuronIndex) {
		// (Network, PositionInNeuronWeightArray) -> Weight
		Weight[][] neuronWeights = new Weight[workers.length][];
		for (int i = 0; i < workers.length; i++) {
			neuronWeights[i] = workers[i].getNeuralNetwork()
					.getLayerAt(layerIndex).getNeuronAt(neuronIndex)
					.getWeights();
		}

		// foreach weight in neuron weight array
		for (int i = 0; i < neuronWeights[0].length; i++) {
			WeightAdjustment adjustment = getWeightAdjustmentForIndex(
					neuronWeights, i);

			setWeightAndWeightChangeAtIndexForAllNets(neuronWeights, i,
					adjustment.adjustedWeight, adjustment.adjustedWeightChange);
		}
	}

	class WeightAdjustment {
		double adjustedWeight, adjustedWeightChange;
	}

	public abstract WeightAdjustment getWeightAdjustmentForIndex(
			Weight[][] neuronWeights, int index);

	private void setWeightAndWeightChangeAtIndexForAllNets(
			Weight[][] neuronWeights, int index, double adjustedWeight,
			double adjustedWeightChange) {
		for (int net = 0; net < workers.length; net++) {
			Weight weight = neuronWeights[net][index];
			weight.value = adjustedWeight;
			weight.weightChange = adjustedWeightChange;
		}
	}
}
