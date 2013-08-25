package wsd.classifiers;

import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;

public class NeuralNetworkClassifier {
	
	private static String[] options = new String[14];
	
	static {
		options[0] = "-L";
		options[1] = "0.3"; 
		options[2] = "-M"; 
		options[3] = "0.2"; 
		options[4] = "-N"; 
		options[5] = "100"; 
		options[6] = "-V"; 
		options[7] = "0"; 
		options[8] = "-S"; 
		options[9] = "0"; 
		options[10] = "-E"; 
		options[11] = "20"; 
		options[12] = "-H"; 
		options[13] = "a"; 
	}

	public static Classifier newInstance() throws Exception {
		MultilayerPerceptron nn = new MultilayerPerceptron();
		nn.setOptions(options);
		return nn;
	}
}
