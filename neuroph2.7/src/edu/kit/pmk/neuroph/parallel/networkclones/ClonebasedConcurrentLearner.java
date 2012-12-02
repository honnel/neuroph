package edu.kit.pmk.neuroph.parallel.networkclones;

import java.io.IOException;
import java.util.concurrent.CyclicBarrier;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.learning.DataSet;
import org.neuroph.core.learning.DataSetRow;

public class ClonebasedConcurrentLearner {
	
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
	public void learnParallel(int numThreads, int syncFrequency,
			NeuralNetwork neuralNet, DataSet dataSet)
			throws InterruptedException {
		NeuralNetwork[] nets = createNeuralNetworkClones(numThreads, neuralNet);

		DataSet[] dataSets = splitDataSet(numThreads, dataSet);

		final NeuralNetInterpolator interpolator = new NeuralNetInterpolator(
				nets);

		final CyclicBarrier barrier = new CyclicBarrier(nets.length,
				new Runnable() {

					@Override
					public void run() {
						interpolator.interpolateWeights();
//						System.out.println("Interpolate. Current time is "
//								+ (System.currentTimeMillis() - t0) + "ms.");
					}
				});

		Thread[] workers = initializeAndStartWorkers(syncFrequency, dataSets,
				nets, barrier);
		waitForWorkersCompletion(workers);
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

	private NeuralNetwork[] createNeuralNetworkClones(int numClones,
			NeuralNetwork neuralNet) {
		NeuralNetwork[] nets = new NeuralNetwork[numClones];
		nets[0] = neuralNet;
		for (int i = 1; i < nets.length; i++) {
			try {
				nets[i] = (NeuralNetwork) DeepCopy.createDeepCopy(neuralNet);
			} catch (ClassNotFoundException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		return nets;
	}

	private void waitForWorkersCompletion(Thread[] workers)
			throws InterruptedException {
		for (int i = 0; i < workers.length; i++) {
			workers[i].join();
		}
	}

	private Thread[] initializeAndStartWorkers(int syncFrequency,
			DataSet[] dataSets, NeuralNetwork[] nets,
			final CyclicBarrier barrier) {
		Thread[] workers = new Thread[nets.length];
		for (int i = 0; i < workers.length; i++) {
			workers[i] = new Thread(new ShadowNetWorker(barrier, nets[i],
					dataSets[i], syncFrequency));
		}
		for (int i = 0; i < workers.length; i++) {
			workers[i].start();
		}
		return workers;
	}
}
