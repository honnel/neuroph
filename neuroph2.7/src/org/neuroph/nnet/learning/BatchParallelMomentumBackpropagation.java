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

import org.neuroph.core.Connection;
import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.Weight;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.learning.DataSet;
import org.neuroph.core.learning.DataSetRow;

/**
 * Backpropagation learning rule with momentum.
 * 
 * @author Zoran Sevarac <sevarac@gmail.com>
 */
public class BatchParallelMomentumBackpropagation extends MomentumBackpropagation {

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
	public BatchParallelMomentumBackpropagation(int threads) {
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
		for (int i = 0; i < threads; i++) {
			workers[i] = new BatchWorker(i, trainingParts[i]);
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
	protected void updateNeuronWeights(Neuron neuron) {
		for (Connection connection : neuron.getInputConnections()) {
			double input = connection.getInput();
			if (input == 0) {
				continue;
			}

			// get the error for specified neuron,
			double neuronError = neuron.getError();

			// tanh can be used to minimise the impact of big error values,
			// which can cause network instability
			// suggested at
			// https://sourceforge.net/tracker/?func=detail&atid=1107579&aid=3130561&group_id=238532
			// double neuronError = Math.tanh(neuron.getError());

			Weight weight = connection.getWeight();
			BatchParallelMomentumWeightAddOn weightTrainingData = (BatchParallelMomentumWeightAddOn) weight.getTrainingData();

			// double currentWeightValue = weight.getValue();
			double previousWeightValue = weightTrainingData.previousValue;
			double weightChange = this.learningRate * neuronError * input + momentum * (weight.value - previousWeightValue);
			// save previous weight value
			// weight.getTrainingData().set(TrainingData.PREVIOUS_WEIGHT,
			// currentWeightValue);

			weight.weightChange += weightChange;
		}
	}

	@Override
	protected void afterEpoch() {

	}

	public double getMomentum() {
		return momentum;
	}

	public void setMomentum(double momentum) {
		this.momentum = momentum;
	}

	@Override
	protected void onStart() {
		// create MomentumWeightTrainingData objects that will be used during
		// the training to store previous weight value
		for (Layer layer : this.neuralNetwork.getLayers()) {
			for (Neuron neuron : layer.getNeurons()) {
				for (Connection connection : neuron.getInputConnections()) {
					connection.getWeight().setTrainingData(new BatchParallelMomentumWeightAddOn());
				}
			} // for
		} // for
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

	public class BatchParallelMomentumWeightAddOn {

		public double weightDelta;
		public double thresholdDelta;

		public double previousValue;
	}

	private class BatchWorker extends Thread {

		private final DataSet trainingSet;
		private final int threadId;
		private NeuralNetwork clone;
		private int currentIteration;

		public BatchWorker(int threadId, DataSet dataSet) {
			this.trainingSet = dataSet;
			this.threadId = threadId;
			currentIteration = 0;
		}

		@Override
		public void run() {
			this.clone = BatchParallelMomentumBackpropagation.this.neuralNetwork.clone();

			while (!isStopped()) {
				beforeEpoch();
				doLearningEpoch(trainingSet);
				this.currentIteration++;
				afterEpoch();

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

				fireLearningEvent(new LearningEvent(BatchParallelMomentumBackpropagation.this)); // notify
																									// listeners

			}
		}
	}
}