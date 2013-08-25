package wsd.app;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import org.junit.Test;

import weka.classifiers.Classifier;
import wsd.utils.TestUtil;

public class Main {
	
	public static void main(String[] args) throws Exception {
		HashMap<String, Classifier> classifiers = loadClassifiers();
		TestUtil.test(classifiers);
	}

	private static HashMap<String, Classifier> loadClassifiers() throws Exception {
		BufferedReader in = new BufferedReader(
								new InputStreamReader(
									new FileInputStream("data/classifier_selector.txt")));
		
		HashMap<String, Classifier> bestClassifiers = new HashMap<String, Classifier>();
		while (true) {
			String line = in.readLine();
			if (line == null) {
				break;
			}
			String[] array = line.split(" ");
			String word = array[0];
			String classifier = array[1];
			bestClassifiers.put(word, ClassifierBuilder.buildClassifier(classifier));
		}
		
		in.close();
		return bestClassifiers;
	}

	@Test
	public void selectBestClassifiers() throws Exception {
		HashMap<String, Classifier> classifiers = ClassifierBuilder.buildClassifiers();
		
		// dump the names of the classifiers into an array
		Set<String> clfNamesSet = classifiers.keySet();
		String[] clfNamesArray = new String[clfNamesSet.size()];
		clfNamesArray = clfNamesSet.toArray(clfNamesArray);
		
		HashMap<String, Double[]> resultsMap = new HashMap<String, Double[]>();
		for (int i = 0; i < clfNamesArray.length; i++) {
			Classifier clf = classifiers.get(clfNamesArray[i]);
			
			System.out.println("evaluating " + clfNamesArray[i]);
			Map<String, Double> resultsPerClf = TestUtil.evaluate(clf);
			Set<String> targetWords = resultsPerClf.keySet();
			
			for (String tw : targetWords) {
				Double[] resultsPerWord = resultsMap.get(tw);
				if (resultsPerWord == null) {
					resultsPerWord = new Double[clfNamesArray.length];
					resultsMap.put(tw, resultsPerWord);
				}
				resultsPerWord[i] = resultsPerClf.get(tw);
			}
		}
		
		// a map between each target word and the name of the best classifier for it
		HashMap<String, String> bestClassifiers = new HashMap<String, String>();
		Set<String> targetWords = resultsMap.keySet();
		for (String tw : targetWords) {
			Double[] resultsPerWord = resultsMap.get(tw);
			
			int posMax = 0;
			double max = 0d;
			for (int i = 0; i < resultsPerWord.length; i++) {
				if (max < resultsPerWord[i]) {
					max = resultsPerWord[i];
					posMax = i;
				}
			}
			
			bestClassifiers.put(tw, clfNamesArray[posMax]);
		}
		
		for (String tw : targetWords) {
			System.out.println(tw + " " + bestClassifiers.get(tw));
		}
	}
}
