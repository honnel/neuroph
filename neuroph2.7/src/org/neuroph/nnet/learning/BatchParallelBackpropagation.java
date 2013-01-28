/**
 * Copyright 2010 Neuroph Project http://neuroph.sourceforge.net
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */
package org.neuroph.nnet.learning;

import java.io.IOException;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;

import org.neuroph.core.Connection;
import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.Weight;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.learning.DataSet;
import org.neuroph.core.learning.DataSetRow;
import org.neuroph.core.learning.SupervisedLearning;

import edu.kit.pmk.neuroph.log.Log;
import edu.kit.pmk.neuroph.parallel.networkclones.FastDeepCopy;

/**
 * Backpropagation learning rule with momentum.
 * 
 */
public class BatchParallelBackpropagation extends BackPropagation {

	/**
	 * The class fingerprint that is set to indicate serialization compatibility
	 * with a previous version of the class.
	 */
	private static final long serialVersionUID = 1L;
	/**
	 * Momentum factor
	 */
	protected double momentum = 0.25d;

	private final int threads;

	/**
	 * Creates new instance of MomentumBackpropagation learning
	 */
	public BatchParallelBackpropagation(int threads) {
		super();
		this.threads = threads;

	}

	@Override
	final public void learn(DataSet trainingSet) {
		setTrainingSet(trainingSet); // set this field here su subclasses can
										// access it
		onStart();

		BatchWorker[] workers = new BatchWorker[threads];
		DataSet[] trainingParts = splitDataSet(threads, trainingSet);
		CyclicBarrier barrier = new CyclicBarrier(threads, new WeightInterpolator(workers));

		CyclicBarrier phase1Barrier = new CyclicBarrier(threads);
		for (int i = 0; i < threads; i++) {
			workers[i] = new BatchWorker(i, trainingParts[i], barrier, workers, phase1Barrier);
			workers[i].start();
		}
		try {
			for (int i = 0; i < threads; i++) {
				workers[i].join();
			}
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}

	/**
	 * This method implements weights update procedure for the single neuron for
	 * the back propagation with momentum factor
	 * 
	 * @param neuron
	 *            neuron to update weights
	 */

	@Override
	protected void afterEpoch() {

	}

	public double getMomentum() {
		return momentum;
	}

	public void setMomentum(double momentum) {
		this.momentum = momentum;
	}

	private static DataSet[] splitDataSet(int numSubsets, DataSet dataSet) {
		DataSet[] dataSets = new DataSet[numSubsets];
		for (int i = 0; i < dataSets.length; i++) {
			dataSets[i] = new DataSet(dataSet.getInputSize(), dataSet.getOutputSize());
		}
		int rowIndex = 0;
		for (DataSetRow row : dataSet.getRows()) {
			dataSets[rowIndex % dataSets.length].addRow(row);
			rowIndex++;
		}
		return dataSets;
	}

	private class BatchWorker extends Thread {

		private final DataSet trainingSet;
		private final int threadId;
		private NeuralNetwork clone;
		private int currentIteration;
		private final CyclicBarrier phase2Barrier;
		private final CyclicBarrier phase1Barrier;

		private Weight[][][] lastWeights;

		private BatchWorker[] colleagues;

		public BatchWorker(int threadId, DataSet dataSet, CyclicBarrier barrier, BatchWorker[] colleagues, CyclicBarrier phase1Barrier) {
			this.trainingSet = dataSet;
			this.threadId = threadId;
			this.phase2Barrier = barrier;
			this.colleagues = colleagues;
			this.phase1Barrier = phase1Barrier;
			currentIteration = 0;
		}

		@Override
		public void run() {
			try {
				this.clone = (NeuralNetwork) FastDeepCopy.createDeepCopy(neuralNetwork);
			} catch (ClassNotFoundException e1) {
				e1.printStackTrace();
				System.exit(1);
			} catch (IOException e) {
				e.printStackTrace();
			}
			this.clone.setLearningRule(new BatchParallelSlave());
			((SupervisedLearning) clone.getLearningRule()).setBatchMode(true);
			((BatchParallelSlave) clone.getLearningRule()).setTrainingSet(trainingSet);
			lastWeights = extractWeights();

			while (!isStopped()) {
				clone.learn(trainingSet);

				// double weight =
				// clone.getLayerAt(1).getNeuronAt(0).getWeights()[0].weightChange;
				// System.out.println("weight: " + weight);

				this.currentIteration++;
				try {
					phase1Barrier.await();
					interpolateNeuronWeights();
					phase2Barrier.await();
				} catch (InterruptedException e) {
					e.printStackTrace();
				} catch (BrokenBarrierException e) {
					e.printStackTrace();
				}

				// todo: abstract stop condition - create abstract class or
				// interface StopCondition
				if (iterationsLimited && (currentIteration >= maxIterations)) {
					stopLearning();
				} else if (!iterationsLimited && (currentIteration < 0)) {
					// restart iteration counter since it has reached max value
					// and
					// iteration number is not limited
					this.currentIteration = 1;
				}

				fireLearningEvent(new LearningEvent(BatchParallelBackpropagation.this)); // notify
			}
			if (threadId == 0) {
				Log.debug(TAG, "Stopped after iterations: " + currentIteration);
			}

		}
		
		private void interpolateNeuronWeights() {
			for (int i = 0; i < neuralNetwork.getLayersCount(); i++) {
				Layer layer = neuralNetwork.getLayerAt(i);
				for (int j = threadId; j < layer.getNeuronsCount(); j += colleagues.length) {
					Neuron neuron = layer.getNeuronAt(j);
					for (int k = 0; k < neuron.getWeights().length; k++) {
						double newValue = neuralNetwork.getLayerAt(i).getNeuronAt(j).getWeights()[k].value;
						for (int w = 0; w < colleagues.length; w++) {
							newValue += colleagues[w].getWeights()[i][j][k].weightChange;
							colleagues[w].getWeights()[i][j][k].weightChange = 0.0;
						}
						neuralNetwork.getLayerAt(i).getNeuronAt(j).getWeights()[k].value = newValue;
						for (int w = 0; w < colleagues.length; w++) {
							colleagues[w].getWeights()[i][j][k].value = newValue;
						}
					}
				}
			}
		}

		//
		private void copyBackNeuronWeights() {
			final Layer[] layers = clone.getLayers();
			for (int i = 1; i < layers.length; i++) {
				final Neuron[] neurons = layers[i].getNeurons();
				for (int j = 0; j < neurons.length; j++) {
					final Weight[] weights = neurons[j].getWeights();
					for (int k = 0; k < weights.length; k++) {
						weights[k].weightChange = 0.0;
						weights[k].value = neuralNetwork.getLayerAt(i).getNeuronAt(j).getWeights()[k].value;
					}
				}
			}
		}

		private Weight[][][] extractWeights() {

			Weight[][][] neuronWeights = new Weight[clone.getLayers().length][][];
			int idx = 0;
			for (Layer l : clone.getLayers()) {
				neuronWeights[idx] = new Weight[l.getNeuronsCount()][];
				for (int i = 0; i < l.getNeuronsCount(); i++) {
					neuronWeights[idx][i] = l.getNeuronAt(i).getWeights();
				}
				idx++;
			}
			return neuronWeights;
		}

		public Weight[][][] getWeights() {
			return this.lastWeights;
		}

		protected double getTotalError() {
			return ((MomentumBackpropagation) clone.getLearningRule()).getTotalNetworkError();
		}
	}

	private final static String TAG = BatchWorker.class.getSimpleName();

	class WeightInterpolator implements Runnable {

		private final BatchWorker[] workers;

		public WeightInterpolator(BatchWorker[] worker) {
			this.workers = worker;
		}

		@Override
		public void run() {
			double totalError = 0.0;
			for (BatchWorker w : workers) {
				totalError += w.getTotalError();
			}
			totalError /= (double) workers.length;
//			System.out.println("fehler: " + totalError);
			if (totalError < BatchParallelBackpropagation.this.maxError) {
				stopLearning();
			}

		}
	}
}