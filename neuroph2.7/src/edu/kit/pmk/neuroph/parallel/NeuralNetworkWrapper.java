package edu.kit.pmk.neuroph.parallel;

import java.io.IOException;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.learning.DataSet;

import edu.kit.pmk.neuroph.log.Log;
import edu.kit.pmk.neuroph.parallel.networkclones.FastDeepCopy;

public class NeuralNetworkWrapper implements ILearner {

	private final String description;
	private NeuralNetwork originalNet;
	private NeuralNetwork neuralNet;
	private int numThreads;

	private static final String TAG = NeuralNetworkWrapper.class
			.getSimpleName();

	public NeuralNetworkWrapper(NeuralNetwork neuralNet, int numThreads,
			String description) {
		this.originalNet = neuralNet;
		this.numThreads = numThreads;
		resetToUnlearnedState();
		this.description = description;
	}

	public NeuralNetworkWrapper(NeuralNetwork neuralNet, int numThreads) {
		this(neuralNet, numThreads, "unnamed");
	}

	@Override
	public void learn(DataSet trainingSet) {
		Log.debug(TAG, "start learning!");
		long t0 = System.currentTimeMillis();
		neuralNet.learn(trainingSet);
		long t1 = System.currentTimeMillis();
		Log.debug(TAG, "learning = " + (t1 - t0) + " ms");
	}

	@Override
	public void resetToUnlearnedState() {
		try {
			this.neuralNet = (NeuralNetwork) FastDeepCopy
					.createDeepCopy(originalNet);
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@Override
	public NeuralNetwork getNeuralNetwork() {
		return neuralNet;
	}

	@Override
	public int getNumberOfThreads() {
		return numThreads;
	}

	public String getDescription() {
		return this.description;
	}

}
