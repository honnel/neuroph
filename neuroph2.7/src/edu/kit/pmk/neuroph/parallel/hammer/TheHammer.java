package edu.kit.pmk.neuroph.parallel.hammer;

import java.io.IOException;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.learning.DataSet;
import org.neuroph.core.learning.DataSetRow;
import org.neuroph.nnet.learning.LMS;

import edu.kit.pmk.neuroph.log.Log;
import edu.kit.pmk.neuroph.parallel.ILearner;
import edu.kit.pmk.neuroph.parallel.networkclones.FastDeepCopy;
import edu.kit.pmk.neuroph.parallel.networkclones.interpolation.NeuralNetInterpolatorType;

public class TheHammer implements ILearner {

	private NeuralNetwork neuralNet;
	private int numThreads;
	private NeuralNetwork originalNet;
	private final String description;

	private static final String TAG = TheHammer.class.getSimpleName();

	public TheHammer(int numThreads, NeuralNetwork neuralNet, String description) {
		this.numThreads = numThreads;
		this.originalNet = neuralNet;
		this.description = description;
		resetToUnlearnedState();
	}

	public TheHammer(int numThreads, int syncFrequency,
			NeuralNetInterpolatorType interpolationType, NeuralNetwork neuralNet) {
		this(numThreads, neuralNet, "");
	}

	class LearnIt implements Callable<Object> {
		DataSet set;

		public LearnIt(DataSet set) {
			this.set = set;
		}

		public Object call() throws Exception {
			neuralNet.learn(set);
			return null;
		}
	}

	@Override
	public void learn(DataSet trainingSet) {
		((LMS) neuralNet.getLearningRule()).setBatchMode(true);
		final DataSet[] dataSets = splitDataSet(numThreads, trainingSet);
		ExecutorService pool = Executors.newFixedThreadPool(numThreads);
		Future<Object>[] futures = new Future[numThreads];

		long t0 = System.currentTimeMillis();
		for (int i = 0; i < numThreads; i++) {
			futures[i] = pool.submit(new LearnIt(dataSets[i]));
		}

		for (int i = 0; i < numThreads; i++) {
			try {
				futures[i].get();
			} catch (InterruptedException | ExecutionException e) {
				e.printStackTrace();
			}
		}
		pool.shutdown();

		long t1 = System.currentTimeMillis();
		Log.debug(TAG, "total = " + (t1 - t0) + " ms");
	}

	private DataSet[] splitDataSet(int numSubsets, DataSet dataSet) {
		DataSet[] dataSets = new DataSet[numSubsets];
		for (int i = 0; i < dataSets.length; i++) {
			dataSets[i] = new DataSet(dataSet.getInputSize(),
					dataSet.getOutputSize());
		}
		int rowIndex = 0;
		for (DataSetRow row : dataSet.getRows()) {
			dataSets[rowIndex % dataSets.length].addRow(row);
			rowIndex++;
		}
		return dataSets;
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

	@Override
	public String getDescription() {
		return description;
	}
}
