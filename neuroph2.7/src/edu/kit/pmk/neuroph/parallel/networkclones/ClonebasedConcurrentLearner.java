package edu.kit.pmk.neuroph.parallel.networkclones;

import java.io.IOException;
import java.util.concurrent.CyclicBarrier;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.learning.DataSet;
import org.neuroph.core.learning.DataSetRow;

import edu.kit.pmk.neuroph.parallel.ILearner;
import edu.kit.pmk.neuroph.parallel.networkclones.FastDeepCopy;
import edu.kit.pmk.neuroph.parallel.networkclones.interpolation.NeuralNetInterpolator;
import edu.kit.pmk.neuroph.parallel.networkclones.interpolation.NeuralNetInterpolatorType;

public class ClonebasedConcurrentLearner implements ILearner {

	private NeuralNetwork neuralNet;
	private int numThreads;
	private int syncFrequency;
	private NeuralNetInterpolatorType interpolationType;
	private CloneNetWorker[] workers;
	private NeuralNetwork originalNet;
	private final String description;

	public ClonebasedConcurrentLearner(int numThreads, int syncFrequency, NeuralNetInterpolatorType interpolationType, NeuralNetwork neuralNet,
			String description) {
		this.numThreads = numThreads;
		this.syncFrequency = syncFrequency;
		this.interpolationType = interpolationType;
		this.originalNet = neuralNet;
		this.description = description;
		resetToUnlearnedState();
	}

	public ClonebasedConcurrentLearner(int numThreads, int syncFrequency, NeuralNetInterpolatorType interpolationType, NeuralNetwork neuralNet) {
		this(numThreads, syncFrequency, interpolationType, neuralNet, "");
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
	@Override
	public void learn(DataSet trainingSet) {
		DataSet[] dataSets = splitDataSet(numThreads, trainingSet);

		final NeuralNetInterpolator interpolator = NeuralNetInterpolator.createNeuralNetInterpolator(interpolationType);
		final CyclicBarrier barrier = new CyclicBarrier(numThreads, new Runnable() {

			@Override
			public void run() {
				interpolator.interpolateWeights();
				// System.out.println("Interpolate. Current time is "
				// + (System.currentTimeMillis() - t0) + "ms.");
			}
		});

		workers = initializeWorkers(syncFrequency, dataSets, barrier);
		interpolator.setWorkers(workers);
		Thread[] threads = startWorkers(workers);
		try {
			waitForWorkersCompletion(threads);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		this.neuralNet = workers[0].getNeuralNetwork();
	}

	public CloneNetWorker[] getCloneNetWorkers() {
		return workers;
	}

	private DataSet[] splitDataSet(int numSubsets, DataSet dataSet) {
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

	private CloneNetWorker[] initializeWorkers(int syncFrequency, DataSet[] dataSets, final CyclicBarrier barrier) {
		CloneNetWorker[] workers = new CloneNetWorker[dataSets.length];
		for (int i = 0; i < workers.length; i++) {
			workers[i] = new CloneNetWorker(barrier, neuralNet, dataSets[i], syncFrequency);
		}
		return workers;
	}

	private Thread[] startWorkers(CloneNetWorker[] workers) {
		Thread[] threads = new Thread[workers.length];
		for (int i = 0; i < threads.length; i++) {
			threads[i] = new Thread(workers[i]);
			threads[i].start();
		}
		return threads;
	}

	private void waitForWorkersCompletion(Thread[] workers) throws InterruptedException {
		for (int i = 0; i < workers.length; i++) {
			workers[i].join();
		}
	}

	@Override
	public void resetToUnlearnedState() {
		try {
			this.neuralNet = (NeuralNetwork) FastDeepCopy.createDeepCopy(originalNet);
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
