package edu.kit.pmk.neuroph.parallel.networkclones;

import java.io.IOException;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.learning.DataSet;
import org.neuroph.core.learning.DataSetRow;

import edu.kit.pmk.neuroph.parallel.networkclones.FastDeepCopy;

public class CloneNetWorker implements Runnable {

	private CyclicBarrier barrier;
	private NeuralNetwork net;
	private DataSet[] runs;

	public CloneNetWorker(CyclicBarrier barrier, NeuralNetwork net, DataSet set, int syncFrequency) {
		this.barrier = barrier;
		this.net = net;
		splitDataSetIntoRuns(set, syncFrequency);
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
//		int counter = 0;
//		for (DataSetRow row : set.getRows()) {
//			runs[i % numRuns].addRow(row);
//			counter++;
//			if(counter==25) {
//				counter = 0;
//				i++;
//			}
//		}
		
	}

	public NeuralNetwork getNeuralNetwork() {
		return this.net;
	}

	public int getNumberOfRuns() {
		return runs.length;
	}

	@Override
	public void run() {
		try {
			this.net = (NeuralNetwork) FastDeepCopy.createDeepCopy(net);
		} catch (ClassNotFoundException | IOException e1) {
			e1.printStackTrace();
		}
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
	}

}