/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package org.neuroph.contrib.samples.timeseries;

import java.util.Arrays;
import java.util.Observable;
import java.util.Observer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.SupervisedLearning;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.core.learning.DataSet;
import org.neuroph.core.learning.DataSetRow;
import org.neuroph.core.learning.LearningRule;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.TransferFunctionType;

/**
 *
 * @author zoran
 */
public class TestTimeSeries implements LearningEventListener {
    NeuralNetwork neuralNet;
    DataSet trainingSet;
    
  public static void main(String[] args) {                
      TestTimeSeries tts = new TestTimeSeries();
      tts.train();
      tts.testNeuralNetwork();     
    }
  
    public void train() {
        // get the path to file with data
        String inputFileName = "C:\\timeseries\\BSW15";
        
        // create MultiLayerPerceptron neural network
        neuralNet = new MultiLayerPerceptron(TransferFunctionType.TANH, 5, 10, 1);
        MomentumBackpropagation learningRule = (MomentumBackpropagation)neuralNet.getLearningRule();
        learningRule.setLearningRate(0.2);
        learningRule.setMomentum(0.5);
        // learningRule.addObserver(this);
        learningRule.addListener(this);        
        
        // create training set from file
         trainingSet = DataSet.createFromFile(inputFileName, 5, 1, "\t");
        // train the network with training set
        neuralNet.learn(trainingSet);         
        
        // add observer here
        
        System.out.println("Done training.");          
    }
  
    
    /**
     * Prints network output for the each element from the specified training set.
     * @param neuralNet neural network
     * @param trainingSet training set
     */
    public void testNeuralNetwork() {
        System.out.println("Testing network...");
        for(DataSetRow trainingElement : trainingSet.getRows()) {
            neuralNet.setInput(trainingElement.getInput());
            neuralNet.calculate();
            double[] networkOutput = neuralNet.getOutput();

            System.out.print("Input: " + Arrays.toString( trainingElement.getInput() ) );
            System.out.println(" Output: " + Arrays.toString( networkOutput) );
        }
    }

//	@Override
//	public void update(Observable arg0, Object arg1) {
//		SupervisedLearning rule = (SupervisedLearning)arg0;
//		System.out.println( "Training, Network Epoch " + rule.getCurrentIteration() + ", Error:" + rule.getTotalNetworkError());
//	}

    @Override
    public void handleLearningEvent(LearningEvent event) {
		SupervisedLearning rule = (SupervisedLearning)event.getSource();
		System.out.println( "Training, Network Epoch " + rule.getCurrentIteration() + ", Error:" + rule.getTotalNetworkError());
    }
    
}
