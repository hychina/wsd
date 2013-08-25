package wsd.classifiers;

import weka.classifiers.Classifier;
import weka.classifiers.functions.Logistic;

public class LogisticClassifier {
	
	private static String[] options = new String[4];
	
	static {
		options[0] = "-R";
		options[1] = "1.0E-8"; 
		options[2] = "-M"; 
		options[3] = "-1"; 
	}

	public static Classifier newInstance() throws Exception {
		Logistic logistic = new Logistic();
		logistic.setOptions(options);
		return logistic;
	}
}
