package edu.kit.pmk.neuroph.parallel.networksiblings;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Weight;

public class NeuralNetInterpolator {

	private SiblingNetWorker[] workers;

	public NeuralNetInterpolator() {
	}

	public void setWorkers(SiblingNetWorker[] workers) {
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
		Weight[][] neuronWeights = new Weight[workers.length][];
		for (int i = 0; i < workers.length; i++) {
			neuronWeights[i] = workers[i].getNeuralNetwork()
					.getLayerAt(layerIndex).getNeuronAt(neuronIndex)
					.getWeights();
		}

		for (int i = 0; i < neuronWeights[0].length; i++) {
			double avgWeight = 0;
			double avgWeightChange = 0;
			for (int net = 0; net < workers.length; net++) {
				avgWeight += neuronWeights[net][i].value;
				avgWeightChange += neuronWeights[net][i].weightChange;
			}
			avgWeight /= workers.length;
			avgWeightChange /= workers.length;
			for (int net = 0; net < workers.length; net++) {
				Weight weight = neuronWeights[net][i];
				weight.value = avgWeight;
				weight.weightChange = avgWeightChange;
			}
		}
	}
}
