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

package org.neuroph.nnet.learning;

import java.util.List;
import org.neuroph.core.Connection;
import org.neuroph.core.Layer;
import org.neuroph.core.Neuron;
import org.neuroph.core.transfer.TransferFunction;

/**
 * Back Propagation learning rule for Multi Layer Perceptron neural networks.
 * 
 * @author Zoran Sevarac <sevarac@gmail.com> 
 * 
 */
public class BackPropagation extends SigmoidDeltaRule {
	
	/**
	 * The class fingerprint that is set to indicate serialization
	 * compatibility with a previous version of the class.
	 */	
	private static final long serialVersionUID = 1L;

	/**
	 * Creates new instance of BackPropagation learning
	 */
	public BackPropagation() {
		super();
	}


	/**
	 * This method implements weight update procedure for the whole network
	 * for the specified  output error vector
	 * 
	 * @param outputError
	 *            output error vector
	 */
	@Override
	protected void updateNetworkWeights(double[] outputError) {
		this.calculateErrorAndUpdateOutputNeurons(outputError); // inherited from SigmoidDeltaRule
		this.calculateErrorAndUpdateHiddenNeurons();            // implemented in this class
	}

	/**
	 * This method implements weights adjustment for the hidden layers
	 */
	protected void calculateErrorAndUpdateHiddenNeurons() {
                Layer[] layers = neuralNetwork.getLayers();
		for (int layerIdx = layers.length - 2; layerIdx > 0; layerIdx--) {	
			for(Neuron neuron : layers[layerIdx].getNeurons()) {	
                                // calculate the neuron's error (delta)
				double neuronError = this.calculateHiddenNeuronError(neuron); 
				neuron.setError(neuronError);
				this.updateNeuronWeights(neuron);
			} // for
		} // for
	}

	/**
	 * Calculates and returns the neuron's error (neuron's delta) for the given neuron param
	 * 
	 * @param neuron
	 *            neuron to calculate error for
	 * @return neuron error (delta) for the specified neuron
	 */
	protected double calculateHiddenNeuronError(Neuron neuron) {		
		double deltaSum = 0d;
		for(Connection connection : neuron.getOutConnections()) {	
			double delta = connection.getToNeuron().getError()
					* connection.getWeight().value;
			deltaSum += delta; // weighted delta sum from the next layer
		} // for

		TransferFunction transferFunction = neuron.getTransferFunction();
		double netInput = neuron.getNetInput(); // should we use input of this or other neuron?
		double f1 = transferFunction.getDerivative(netInput);
		double neuronError = f1 * deltaSum;
		return neuronError;
	}

}
