package edu.kit.pmk.neuroph.parallel.networkclones.revised;

import java.io.IOException;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.learning.DataSet;
import org.neuroph.nnet.learning.LMS;

import edu.kit.pmk.neuroph.parallel.networkclones.CloneNetWorker;
import edu.kit.pmk.neuroph.parallel.networkclones.FastDeepCopy;

public class CloneNetWorkerRevised extends CloneNetWorker implements Runnable {

	private static final String TAG = CloneNetWorkerRevised.class
			.getSimpleName();
	private CyclicBarrier barrier;
	private NeuralNetwork net;
	private DataSet set;
	private int maxIterations;
	private boolean finished;

	public CloneNetWorkerRevised(CyclicBarrier barrier, NeuralNetwork net,
			DataSet set, int maxIterations) {
		super(barrier, net, set);
		this.barrier = barrier;
		this.net = net;
		this.set = set;
		this.maxIterations = maxIterations;
		this.finished = false;
	}
	
	@Override
	public NeuralNetwork getNeuralNetwork() {
		return this.net;
	}


	@Override
	public int getNumberOfRuns() {
		return -1;
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

		LMS learningRule = ((LMS) net.getLearningRule());
		learningRule.setMaxIterations(1);

		long t2 = System.currentTimeMillis();
		for (int i = 0; i < maxIterations; i++) {
			if(!finished) {
				net.learn(set);
			}
			try {
				barrier.await();
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (BrokenBarrierException e) {
				e.printStackTrace();
			}
			if(learningRule.hasReachedStopCondition())
				finished = true;
		}
		long t3 = System.currentTimeMillis();
		System.out.println(TAG + id + ": deepcopy = " + (t1 - t0) + " ms");
		System.out.println(TAG + id + ": learning = " + (t3 - t2) + " ms");
		System.out.println(TAG + id + ": deepcopy + learning = " + (t3 - t0)
				+ " ms");
	}

}