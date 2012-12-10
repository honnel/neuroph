package edu.kit.pmk.neuroph.parallel.networksiblings;

import java.util.concurrent.CyclicBarrier;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.learning.DataSet;
import org.neuroph.core.learning.DataSetRow;

public class SiblingBasedConcurrentLearner {

	private NeuralNetwork neuralNet;
	private DataSet dataSet;
	private int numThreads;
	private int syncFrequency;
	private SiblingNetWorker[] workers;

	public SiblingBasedConcurrentLearner(NeuralNetwork neuralNet,
			DataSet dataSet, int numThreads, int syncFrequency) {
		this.neuralNet = neuralNet;
		this.dataSet = dataSet;
		this.numThreads = numThreads;
		this.syncFrequency = syncFrequency;

		DataSet[] dataSets = splitDataSet(numThreads, dataSet);

		final NeuralNetInterpolator interpolator = new NeuralNetInterpolator();
		final CyclicBarrier barrier = new CyclicBarrier(numThreads,
				new Runnable() {

					@Override
					public void run() {
						interpolator.interpolateWeights();
						// System.out.println("Interpolate. Current time is "
						// + (System.currentTimeMillis() - t0) + "ms.");
					}
				});
		workers = initializeWorkers(syncFrequency, dataSets, barrier);
		interpolator.setWorkers(workers);
	}

	/**
	 * Trains the network with <tt>numThreads</tt> threads concurrently. The
	 * <tt>dataSet</tt> is split into sets of equal size. Each thread trains a
	 * clone of <tt>neuralNet</tt> with its subset of <tt>dataSet</tt>. After
	 * <tt>syncFrequency</tt> training elements the threads synchronize and
	 * interpolate their neural network clones. Interpolation means that each
	 * weight and weightChange property is assigned the average weight and
	 * average weightChange from the neural network clones.
	 * 
	 * @param numThreads
	 * @param syncFrequency
	 *            determines how many training elements are processed before an
	 *            interpolation occurs
	 * @param neuralNet
	 *            the neural network to be trained
	 * @param dataSet
	 * @throws InterruptedException
	 */
	public void learn() throws InterruptedException {
		Thread[] threads = startWorkers(workers);
		waitForWorkersCompletion(threads);
	}

	public SiblingNetWorker[] getCloneNetWorkers() {
		return workers;
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

	private SiblingNetWorker[] initializeWorkers(int syncFrequency,
			DataSet[] dataSets, final CyclicBarrier barrier) {
		SiblingNetWorker[] workers = new SiblingNetWorker[dataSets.length];
		for (int i = 0; i < workers.length; i++) {
			workers[i] = new SiblingNetWorker(barrier, neuralNet, dataSets[i],
					syncFrequency);
		}
		return workers;
	}

	private Thread[] startWorkers(SiblingNetWorker[] workers) {
		Thread[] threads = new Thread[workers.length];
		for (int i = 0; i < threads.length; i++) {
			threads[i] = new Thread(workers[i]);
			threads[i].start();
		}
		return threads;
	}

	private void waitForWorkersCompletion(Thread[] workers)
			throws InterruptedException {
		for (int i = 0; i < workers.length; i++) {
			workers[i].join();
		}
	}
}
