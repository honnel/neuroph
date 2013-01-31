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

package org.neuroph.nnet;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Stack;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.neuroph.core.Layer;
import org.neuroph.core.transfer.Linear;
import org.neuroph.nnet.comp.neuron.BiasNeuron;
import org.neuroph.nnet.comp.neuron.InputNeuron;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.ConnectionFactory;
import org.neuroph.util.NeuralNetworkFactory;
import org.neuroph.util.NeuralNetworkType;
import org.neuroph.util.NeuronProperties;
import org.neuroph.util.random.NguyenWidrowRandomizer;

/**
 * Multi Layer Perceptron neural network with Back propagation learning
 * algorithm.
 * 
 * @see org.neuroph.nnet.learning.BackPropagation
 * @see org.neuroph.nnet.learning.MomentumBackpropagation
 * @author Zoran Sevarac <sevarac@gmail.com>
 */
public class ParallelMultiLayerPerceptron extends MultiLayerPerceptron {
	
	private static final int THREADS = 2;
	
	public ParallelMultiLayerPerceptron(int... neuronsInLayers) {
		super(neuronsInLayers);
	}

	/**
	 * Creates MultiLayerPerceptron Network architecture - fully connected feed
	 * forward with specified number of neurons in each layer
	 * 
	 * @param neuronsInLayers
	 *            collection of neuron numbers in getLayersIterator
	 * @param neuronProperties
	 *            neuron properties
	 */
	protected void createNetwork(List<Integer> neuronsInLayers, NeuronProperties neuronProperties) {

		// Dome: store neuronProperties, needed for exact clone()
		setNeuronProperties(neuronProperties);

		// set network type
		this.setNetworkType(NeuralNetworkType.MULTI_LAYER_PERCEPTRON);

		// create input layer
		NeuronProperties inputNeuronProperties = new NeuronProperties(InputNeuron.class, Linear.class);
		Layer layer = new ParallelLayer(neuronsInLayers.get(0), neuronProperties);

		boolean useBias = true; // use bias neurons by default
		if (neuronProperties.hasProperty("useBias")) {
			useBias = (Boolean) neuronProperties.getProperty("useBias");
		}

		if (useBias) {
			layer.addNeuron(new BiasNeuron());
		}

		this.addLayer(layer);

		// create layers
		Layer prevLayer = layer;

		// for(Integer neuronsNum : neuronsInLayers)
		for (int layerIdx = 1; layerIdx < neuronsInLayers.size(); layerIdx++) {
			Integer neuronsNum = neuronsInLayers.get(layerIdx);
			// createLayer layer
			layer = new ParallelLayer(neuronsNum, neuronProperties);

			if (useBias && (layerIdx < (neuronsInLayers.size() - 1))) {
				layer.addNeuron(new BiasNeuron());
			}

			// add created layer to network
			this.addLayer(layer);
			// createLayer full connectivity between previous and this layer
			if (prevLayer != null) {
				ConnectionFactory.fullConnect(prevLayer, layer);
			}

			prevLayer = layer;
		}

		// set input and output cells for network
		NeuralNetworkFactory.setDefaultIO(this);

		// set learnng rule
		// this.setLearningRule(new BackPropagation(this));
		this.setLearningRule(new MomentumBackpropagation());
		// this.setLearningRule(new DynamicBackPropagation());

		this.randomizeWeights(new NguyenWidrowRandomizer(-0.7, 0.7));

	}

	private transient static ExecutorService service = null;

	private static ExecutorService getExecutor() {
		if (service == null) {
			service = Executors.newCachedThreadPool();
		}
		return service;
	}

	class ParallelLayer extends Layer {

		private static final long serialVersionUID = -6469387025209211369L;

		private final List<NeuronJob> jobs = new ArrayList<NeuronJob>();

		public ParallelLayer(int neuronCount, NeuronProperties prop) {
			super(neuronCount, prop);
			int neuronsPerJob = neuronCount / THREADS;
			if (neuronsPerJob * THREADS < neuronCount) {
				neuronsPerJob++;
			}

			for (int i = 0; i < THREADS; i++) {
				final int from = i * neuronsPerJob;
				final int to = Math.min(from + neuronsPerJob, neuronCount);
				jobs.add(new NeuronJob(from, to));
			}

		}

		@Override
		public void calculate() {
			ExecutorService service = getExecutor();
			Stack<Future<?>> futures = new Stack<>();

			for (NeuronJob j : jobs) {
				futures.push(service.submit(j));
			}
			try {
				while (!futures.isEmpty()) {
					futures.pop().get();
				}
			} catch (InterruptedException | ExecutionException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

		}

		private class NeuronJob implements Runnable, Serializable {

			private static final long serialVersionUID = -1060225293085055790L;

			private final int to;
			private final int from;

			public NeuronJob(int from, int to) {
				this.to = to;
				this.from = from;
			}

			@Override
			public void run() {
				for (int i = from; i < to; i++) {
					neurons[i].calculate();
				}
			}

		}

	}
}