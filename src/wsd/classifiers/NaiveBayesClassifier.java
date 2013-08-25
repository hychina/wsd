package wsd.classifiers;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;

public class NaiveBayesClassifier {
	
	private static String[] options;
	
	static {
		options = new String[1];
		options[0] = "-K";
	}

	public static Classifier newInstance() throws Exception {
		NaiveBayes nb = new NaiveBayes();
		nb.setOptions(options);
		return nb;
	}

}
