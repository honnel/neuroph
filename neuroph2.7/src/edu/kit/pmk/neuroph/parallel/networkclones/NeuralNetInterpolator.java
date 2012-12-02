package edu.kit.pmk.neuroph.parallel.networkclones;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Weight;

public class NeuralNetInterpolator {

	private NeuralNetwork[] nets;

	public NeuralNetInterpolator(NeuralNetwork[] nets) {
		this.nets = nets;
	}

	public void interpolateWeights() {
		for (int layer = 0; layer < nets[0].getLayersCount(); layer++) {
			for (int neuron = 0; neuron < nets[0].getLayerAt(layer)
					.getNeuronsCount(); neuron++) {
				interpolateWeightAndWeightChangeForNeuron(layer, neuron);
			}
		}
	}

	private void interpolateWeightAndWeightChangeForNeuron(int layerIndex,
			int neuronIndex) {
		Weight[][] neuronWeights = new Weight[nets.length][];
		for (int i = 0; i < nets.length; i++) {
			neuronWeights[i] = nets[i].getLayerAt(layerIndex)
					.getNeuronAt(neuronIndex).getWeights();
		}

		for (int i = 0; i < neuronWeights[0].length; i++) {
			double avgWeight = 0;
			double avgWeightChange = 0;
			for (int net = 0; net < nets.length; net++) {
				avgWeight += neuronWeights[net][i].value;
				avgWeightChange += neuronWeights[net][i].weightChange;
			}
			avgWeight /= nets.length;
			avgWeightChange /= nets.length;
			for (int net = 0; net < nets.length; net++) {
				Weight weight = neuronWeights[net][i];
				weight.value = avgWeight;
				weight.weightChange = avgWeightChange;
			}
		}
	}
}
