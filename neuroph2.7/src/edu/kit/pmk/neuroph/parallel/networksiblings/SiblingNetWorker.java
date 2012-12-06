package edu.kit.pmk.neuroph.parallel.networksiblings;

import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.learning.DataSet;
import org.neuroph.core.learning.DataSetRow;

class SiblingNetWorker implements Runnable {

	private CyclicBarrier barrier;
	private NeuralNetwork originalNet;
	private NeuralNetwork net;
	private DataSet[] runs;

	public SiblingNetWorker(CyclicBarrier barrier, NeuralNetwork net, DataSet set, int syncFrequency) {
		this.barrier = barrier;
		this.originalNet = net;
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
		this.net = originalNet.clone();
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