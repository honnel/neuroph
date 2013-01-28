package edu.kit.pmk.neuroph.parallel.networkclones;

import java.io.IOException;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.learning.DataSet;
import org.neuroph.core.learning.DataSetRow;

import edu.kit.pmk.neuroph.log.Log;
import edu.kit.pmk.neuroph.parallel.networkclones.FastDeepCopy;

public class CloneNetWorker implements Runnable {

	private static final String TAG = CloneNetWorker.class.getSimpleName();
	private CyclicBarrier barrier;
	private NeuralNetwork net;
	private DataSet[] runs;

	public CloneNetWorker(CyclicBarrier barrier, NeuralNetwork net,
			DataSet set, int syncFrequency) {
		this.barrier = barrier;
		this.net = net;
		splitDataSetIntoRuns(set, syncFrequency);
	}

	// interpolate only at the end
	public CloneNetWorker(CyclicBarrier barrier, NeuralNetwork net, DataSet set) {
		this.barrier = barrier;
		this.net = net;
	}

	private void splitDataSetIntoRuns(DataSet set, int syncFrequency) {
		int numRuns = (set.size() / syncFrequency) + 1;
		this.runs = new DataSet[numRuns];
		for (int k = 0; k < runs.length; k++)
			runs[k] = new DataSet(set.getInputSize(), set.getOutputSize());
		int i = 0;

		// gute aufteilung
		for (DataSetRow row : set.getRows()) {
			runs[i % numRuns].addRow(row);
			i++;
		}

		// boese aufteilung
		// int counter = 0;
		// for (DataSetRow row : set.getRows()) {
		// runs[i % numRuns].addRow(row);
		// counter++;
		// if(counter==25) {
		// counter = 0;
		// i++;
		// }
		// }

	}

	public NeuralNetwork getNeuralNetwork() {
		return this.net;
	}

	public int getNumberOfRuns() {
		return runs.length;
	}

	@Override
	public void run() {
		String id = "[Thread " + Thread.currentThread().getId() + "]";
		long t0 = System.currentTimeMillis();
		try {
			this.net = (NeuralNetwork) FastDeepCopy.createDeepCopy(net);
		} catch (IOException e1) {
			e1.printStackTrace();
		} catch (ClassNotFoundException e1) {
			e1.printStackTrace();
		}
		long t1 = System.currentTimeMillis();
		for (int i = 0; i < runs.length; i++) {
			net.learn(runs[i]);
			try {
				barrier.await();
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (BrokenBarrierException e) {
				e.printStackTrace();
			}
		}
		long t2 = System.currentTimeMillis();
		Log.debug(TAG + id, "deepcopy = " + (t1 - t0) + " ms");
		Log.debug(TAG + id,  "learning = " + (t2 - t1) + " ms");
		Log.debug(TAG + id, "deepcopy + learning = " + (t2 - t0)
				+ " ms");
	}

}