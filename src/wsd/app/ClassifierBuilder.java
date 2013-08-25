package wsd.app;

import java.util.HashMap;

import weka.classifiers.Classifier;
import wsd.classifiers.DecisionTreeClassifier;
import wsd.classifiers.LogisticClassifier;
import wsd.classifiers.NaiveBayesClassifier;
import wsd.classifiers.NeuralNetworkClassifier;

public class ClassifierBuilder {
	
	private static final String[] names = {"weka.classifiers.bayes.NaiveBayes"};
//												 "weka.classifiers.bayes.NaiveBayes"}; 
//												 "weka.classifiers.functions.Logistic",
//												 "weka.classifiers.trees.J48"};
		
	public static HashMap<String, Classifier> buildClassifiers() throws Exception {
		HashMap<String, Classifier> classifiers = new HashMap<String, Classifier>();
		for (String name : names) {
			classifiers.put(name, buildClassifier(name));
		}
		return classifiers;
	}
	
	public static Classifier buildClassifier(String name) throws Exception {
		Classifier clf = null;
		switch (name) {
		case "weka.classifiers.functions.MultilayerPerceptron":
			clf = NeuralNetworkClassifier.newInstance();
			break;
		case "weka.classifiers.functions.Logistic":
			clf = LogisticClassifier.newInstance();
			break;
		case "weka.classifiers.bayes.NaiveBayes":
			clf = NaiveBayesClassifier.newInstance();
			break;
		case "weka.classifiers.trees.J48":
			clf = DecisionTreeClassifier.newInstance();
			break;
		}
		return clf;
	}

}
