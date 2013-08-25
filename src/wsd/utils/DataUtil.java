package wsd.utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeMap;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;


import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instances;
import weka.core.SparseInstance;
import wsd.beans.Example;

public class DataUtil {

	private static final String BOS = "bos";
	private static final String EOS = "eos";
	private static FastVector posTags = new FastVector();
	// record which words have been chosen to be features, and their indexes
	private static Map<String, HashMap<String, Integer>> wordFeaturesByTargetWords = new HashMap<String, HashMap<String, Integer>>();
	// record the answers for the test data
	private static Map<String, HashMap<Integer, String>> answers = new HashMap<String, HashMap<Integer, String>>();
	
	// define the pos tags
	static {
		String posString = "dummy Ag a ad an b c Dg d e f g h i j k l m Ng " +
				"n nr ns nt nz na nx ne o p q r s Tg t u Vg v vd vn w x y z bos eos";
		String[] posArray = posString.split(" ");
		

		for (String pos : posArray) {
			posTags.addElement(pos);
		}
	}
	
	/**
	 * build data sets for training
	 */
	public static Map<String, Instances> buildTrainingDataset(String trainingFile) 
		throws FileNotFoundException, SAXException, IOException, ParserConfigurationException {
		
		Map<String, Instances> datasetByTargetWord = new HashMap<String, Instances>();
		
		// parse the xml file to get the training data
		Map<String, Map<String, List<Example>>> trainingData = parseXml(trainingFile);
		// in order to output the weights of the context words
		PrintWriter out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(
				new FileOutputStream("data/weight.txt"), Charset.forName("utf-8"))));
		// read the stop words into a HashSet
		HashSet<String> stopwords = readStopwords("data/stopwords.txt");		
		
		Set<String> targetWords = trainingData.keySet(); 
		for (String tw : targetWords) {
			Map<String, List<Example>> examplesBySense = trainingData.get(tw);
			HashMap<String, Integer> wordFeatures = new HashMap<String, Integer>();
			wordFeaturesByTargetWords.put(tw, wordFeatures);

			// define the attributes (features)
			Instances dataset = defineDataset(examplesBySense, 
											  tw, 
											  out, 
											  wordFeatures, 
											  stopwords);
			
			// add data
			addData(dataset, examplesBySense, tw, false);

			dataset.setClassIndex(dataset.numAttributes() - 1);
			dataset.compactify();
			datasetByTargetWord.put(tw, dataset);
		}

		out.close();

		return datasetByTargetWord;
	}
	
	/**
	 * build data sets for testing
	 */
	public static Map<String, Instances> buildTestingDataset(String testingFile, Map<String, Instances> trainingDatasetByTargetWord) 
		throws FileNotFoundException, SAXException, IOException, ParserConfigurationException {
		
		Map<String, Instances> testDatasetByTargetWord = new HashMap<String, Instances>();
		// parse the xml file to get the testing data
		Map<String, Map<String, List<Example>>> testingData = parseXml(testingFile);
		// read the answers
		readTestAnswer("data/ChineseLS.test.key");	
				
		Set<String> targetWords = testingData.keySet();
		for (String tw : targetWords) {
			Instances trainingInstances = trainingDatasetByTargetWord.get(tw);
			Instances tesingInstances = new Instances(trainingInstances, 0);
			Map<String, List<Example>> examplesBySense = testingData.get(tw);
			
			addData(tesingInstances, examplesBySense,tw, true);
			tesingInstances.compactify();
			testDatasetByTargetWord.put(tw, tesingInstances);
		}
		return testDatasetByTargetWord;
	}
	
	/**
	 * read the answers for the test data
	 * @return 
	 */
	private static void readTestAnswer(String keyFile) throws IOException {
		BufferedReader in = new BufferedReader(
								new InputStreamReader(
									new FileInputStream(keyFile)));
		
		while (true) {
			String line = in.readLine();
			if (line == null) {
				break;
			}
			String[] lineArray = line.split(" ");
			String idStr = lineArray[1].trim();
			String targetWord = idStr.substring(0, idStr.indexOf("."));
			int id = Integer.parseInt(idStr.substring(idStr.indexOf(".") + 1));
			String sense = lineArray[2].trim();
			
			if (answers.containsKey(targetWord)) {
				HashMap<Integer, String> ansById = answers.get(targetWord);
				ansById.put(id, sense);
			}
			else {
				HashMap<Integer, String> ansById = new HashMap<Integer, String>();
				ansById.put(id, sense);
				answers.put(targetWord, ansById);
			}
		}
		
		in.close();
	}
	
	/**
	 * read the file containing the stop words, and record them in a HashSet
	 */
	private static HashSet<String> readStopwords(String fileName) throws IOException {
		BufferedReader in = new BufferedReader(
								new InputStreamReader(
									new FileInputStream(fileName))); 
		
		HashSet<String> stopwords = new HashSet<String>();
		
		while (true) { 
			String stopword = in.readLine();
			if (stopword == null) 
				break;
			stopwords.add(stopword);
		}
		
		in.close();
		return stopwords;
	}
	
	/**
	 * define the features
	 */
	private static Instances defineDataset(Map<String, List<Example>> examplesBySense, 
										   String targetWord,
										   PrintWriter out,
										   HashMap<String, Integer> wordFeatures,
										   HashSet<String> stopwords) {
		
		Set<String> senses = examplesBySense.keySet();
		// record the weights of the context words in different senses
		Map<String, Map<String, Double>> termWeightsBySense = new HashMap<String, Map<String, Double>>();
		// number of senses a context word appears in
		Map<String, Integer> termSenseCount = new HashMap<String, Integer>(); 
		
		// add a dummy value so that the first sense won't be lost in a sparse instance
		FastVector senseLabels = new FastVector();
		senseLabels.addElement("dummy");
		
		// count number of times a context word appears under each word sense
		for (String sense : senses) {
			// in order to set the sense attribute
			senseLabels.addElement(sense);
			
			List<Example> examples = examplesBySense.get(sense);
			// record the weights of the context words under each sense
			Map<String, Double> termWeights = new HashMap<String, Double>();
			
			for (Example exp : examples) {
				List<String> contextWords = exp.getContextWords();
				
				for (String word : contextWords) {
					Double weight = termWeights.get(word);
					
					if (weight != null) {
						termWeights.put(word, weight + 1d);
					}
					// first time a context word appears under a word sense
					else {
						termWeights.put(word, 1d);
						Integer count = termSenseCount.get(word);
						
						if (count != null) {
							termSenseCount.put(word, count + 1);
						}
						else {
							termSenseCount.put(word, 1);
						}
					}
				}
			}
			
			termWeightsBySense.put(sense, termWeights);
		}
		
		// update the term weights, take into account the idf of each word
		for (String sense : senses) {
			Map<String, Double> termWeights = termWeightsBySense.get(sense);
			Set<String> contextWords = termWeights.keySet();
			for (String word : contextWords) {
				Integer count = termSenseCount.get(word);
				double idf = Math.log((double) senses.size() / (double) count) + 1d;
				termWeights.put(word, termWeights.get(word) * idf);
			}
		}
		
		// set the attributes
		FastVector attributes = new FastVector();
		
		// add the words with weights equal with or higher than 2.0 as features
		int index = 0;
		for (String sense : senses) {
			Map<String, Double> weights = termWeightsBySense.get(sense);
			List<Entry<String, Double>> weightsDesc = SortUtil.sortMapDesc(weights);
			int count = 1;
			
			for (Entry<String, Double> entry : weightsDesc) {
				String word = entry.getKey();
				Double weight = entry.getValue();
				
				if (count > 1 || weight < 2.0d)
					break;
				
				if (!wordFeatures.containsKey(word) 
					&& !stopwords.contains(word)) {
					
					wordFeatures.put(word, index);
					attributes.addElement(new Attribute("word_" + index));
					count++;
					index++;
				}
			}
		}
		
		Attribute posBeforeAttr = new Attribute("pos_before", posTags);
		Attribute posAfterAttr = new Attribute("pos_after", posTags);
		attributes.addElement(posBeforeAttr);
		attributes.addElement(posAfterAttr);
		Attribute senseAttr = new Attribute("sense", senseLabels);
		attributes.addElement(senseAttr);
		
		// output the term weights
		out.println(targetWord);
		out.println("--------------------------------------------------");
				
		for (String sense : senses) {
			out.println(sense);
			
			Map<String, Double> termWeigts = termWeightsBySense.get(sense);
			List<Entry<String, Double>> sortedTermWeights = SortUtil.sortMapDesc(termWeigts);
			
			for (Entry<String, Double> e : sortedTermWeights) {
				out.print(e.getKey() + ":" + e.getValue() + " ");
			}
			out.println();
			out.println();
		}
		// done output the term weights

		return new Instances(targetWord, attributes, 0);
	}
	
	/**
	 * add data to dataset
	 */
	private static void addData(Instances dataset, 
								Map<String, List<Example>> examplesBySense,
								String targetWord,
								boolean isTestData) {
		
		HashMap<String, Integer> wordFeatures = wordFeaturesByTargetWords.get(targetWord);
		Set<String> senses = examplesBySense.keySet();
		for (String sense : senses) {
			List<Example> examples = examplesBySense.get(sense);

			for (Example exp : examples) {
				String posBefore = exp.getPosBefore();
				String posAfter = exp.getPosAfter();
				int id = exp.getId();
				List<String> contextWords = exp.getContextWords();

				// use TreeMap to sort the indices of the values by ascending order (required by weka)
				
				TreeMap<Integer, String> wordAppears = new TreeMap<Integer, String>();
				for (String word : contextWords) {
					if (wordFeatures.containsKey(word)) {
						wordAppears.put(wordFeatures.get(word), word);
					}
				}

				int numWordAppears = wordAppears.size();
				int numOtherAttr = 3;
								
				double values[] = new double[numWordAppears + numOtherAttr];
				int indices[] = new int[numWordAppears + numOtherAttr];

				int i = 0;
				Iterator<Integer> it = wordAppears.keySet().iterator();
				while (it.hasNext()) {
					values[i] = 1;
					indices[i] = it.next();
					i++;
				}

				indices[i] = dataset.numAttributes() - 3;
				indices[i + 1] = dataset.numAttributes() - 2;
				indices[i + 2] = dataset.numAttributes() - 1;
				values[i] = dataset.attribute(indices[i]).indexOfValue(posBefore);
				values[i + 1] = dataset.attribute(indices[i + 1]).indexOfValue(posAfter);
				
				if (!isTestData) {
					values[i + 2] = dataset.attribute(indices[i + 2]).indexOfValue(sense);
				}
				else {
					HashMap<Integer, String> ansById = answers.get(targetWord);
					String wordSense = ansById.get(id);
					values[i + 2] = dataset.attribute(indices[i + 2]).indexOfValue(wordSense);
				}

				SparseInstance sparseInstance = new SparseInstance(1.0d, values, indices, dataset.numAttributes());
				dataset.add(sparseInstance);
			}
		}
	}
	
	/**
	 * parse the given XML file to extract the examples for each target word,
	 * used both for training data and test data
	 */
	private static Map<String, Map<String, List<Example>>> parseXml(String fileName) 
		throws FileNotFoundException, SAXException, IOException, ParserConfigurationException {
		
		DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
		DocumentBuilder db = dbf.newDocumentBuilder();
		Document document = db.parse(new File(fileName));
		
		// trainingData records training examples for each 
		// target word organized by different senses 
		LinkedHashMap<String, Map<String, List<Example>>> trainingData = new LinkedHashMap<String, Map<String, List<Example>>>();
		NodeList targetWords = document.getElementsByTagName("lexelt");
		
		// iterate through the target words
		for (int i = 0; i < targetWords.getLength(); i++) {
			Element targetWord = (Element) targetWords.item(i);
			// the string representing the current target word
			String targetWordStr = targetWord.getAttribute("item");
			// the training examples for each target word
			NodeList instances = targetWord.getElementsByTagName("instance");
			// organize the training examples by word sense
			Map<String, List<Example>> examplesBySense = new HashMap<String, List<Example>>();
			
			// iterate through the training examples
			for (int j = 0; j < instances.getLength(); j++) {
				// the current training example
				Element instance = (Element) instances.item(j);
				// record the current training example
				Example example = new Example();
				// the word sense of the current training example
				String sense = ((Element) instance.getElementsByTagName("answer").item(0)).getAttribute("senseid");
				// the id of the current example
				String idStr = ((Element) instance.getElementsByTagName("answer").item(0)).getAttribute("instance");
				int id = Integer.parseInt(idStr.substring(idStr.indexOf(".") + 1));
				example.setId(id);
				// get the context words
				NodeList contextWords = ((Element) instance.getElementsByTagName("postagging").item(0)).getElementsByTagName("word");
				
				// iterate through the context words
				for (int m = 0; m < contextWords.getLength(); m++) {
					Element wordElement = (Element) contextWords.item(m);
					NodeList subwordElements = wordElement.getElementsByTagName("subword");

					// if the current word element has subwords
					if (subwordElements.getLength() != 0) {
						for (int n = 0; n < subwordElements.getLength(); n++) {
							handleContextWord((Element) subwordElements.item(n), targetWordStr, example);
						}
					}
					else {
						handleContextWord(wordElement, targetWordStr, example);
					}
				}
				
				// the first time a word sense appears
				if (!examplesBySense.containsKey(sense)) {
					// record the training examples of each sense of each target word
					List<Example> examples = new ArrayList<Example>();
					examples.add(example);
					examplesBySense.put(sense, examples);
				} 
				// the word sense already exists
				else {
					examplesBySense.get(sense).add(example);
				}
			}
			
			trainingData.put(targetWordStr, examplesBySense);
		}
		
		return trainingData;
	}
	
	private static void handleContextWord(Element currentWord, 
										  String targetWord, 
										  Example example) {
		
		// the pos of the current word
		String pos = currentWord.getAttribute("pos");
		// the token representing the current word
		String token = ((Element) currentWord.getElementsByTagName("token").item(0))
			.getFirstChild().getTextContent().trim();
		// id of the current word/subword element
		Integer id = Integer.parseInt(currentWord.getAttribute("id"));
		// node name of the current word element
		String nodeName = currentWord.getNodeName();
		
		// if the current word is the target word itself
		if (token.equals(targetWord)) {
			Element parentElement = (Element) currentWord.getParentNode();
			
			// if the current element is of type word
			if (nodeName.equals("word")) {
				NodeList contextWords = parentElement.getElementsByTagName("word");
				
				// if this is the first word element in the context
				if (id.equals(0)) {
					example.setPosBefore(BOS);
				}
				else {
					Element wordBefore = (Element) contextWords.item(id - 1);
					NodeList subwordsBefore = wordBefore.getElementsByTagName("subword");
					
					if (subwordsBefore.getLength() == 0) {
						example.setPosBefore(wordBefore.getAttribute("pos"));
					}
					// if the word element before has subwords, take the pos 
					// of the last subword as the pos before the current word
					else {
						Element subwordBefore = (Element) subwordsBefore.item(subwordsBefore.getLength() - 1);
						example.setPosBefore(subwordBefore.getAttribute("pos"));
					}
				}
				
				// if this is the last word element in the context
				if (id.equals(contextWords.getLength() - 1)) {
					example.setPosAfter(EOS);
				}
				else {
					Element wordAfter = (Element) contextWords.item(id + 1);
					NodeList subwordsAfter = wordAfter.getElementsByTagName("subword");
					
					if (subwordsAfter.getLength() == 0) {
						example.setPosAfter(wordAfter.getAttribute("pos"));
					}
					else {
						Element subwordAfter = (Element) subwordsAfter.item(0);
						example.setPosAfter(subwordAfter.getAttribute("pos"));
					}
				}
			} 
			// if the current element is of type subword
			else if (nodeName.equals("subword")) {
				NodeList contextSubwords = parentElement.getElementsByTagName("subword");
				Integer parentId = Integer.parseInt(parentElement.getAttribute("id"));
				Element grandParent = (Element) parentElement.getParentNode();
				
				// if this is the first subword
				if (id.equals(0)) {
					// if this is the first subword of the first word
					if (parentId.equals(0)) {
						example.setPosBefore(BOS);
					}
					else {
						Element wordBefore = (Element) grandParent.getElementsByTagName("word").item(parentId - 1);
						NodeList subwordsBefore = wordBefore.getElementsByTagName("subword");
						
						if (subwordsBefore.getLength() == 0) {
							example.setPosBefore(wordBefore.getAttribute("pos"));
						}
						else {
							Element subwordBefore = (Element) subwordsBefore.item(subwordsBefore.getLength() - 1);
							example.setPosBefore(subwordBefore.getAttribute("pos"));
						}
					}
				}
				else {
					Element subwordBefore = (Element) contextSubwords.item(id - 1);
					example.setPosBefore(subwordBefore.getAttribute("pos"));
				}
				
				// if this is the last subword
				if (id.equals(contextSubwords.getLength() - 1)) {
					NodeList contextWords = grandParent.getElementsByTagName("word");
					
					// if this is the last subword of the last word
					if (parentId.equals(contextWords.getLength() - 1)) {
						example.setPosAfter(EOS); 
					}
					else {
						Element wordAfter = (Element) contextWords.item(parentId + 1);
						NodeList subwordsAfter = wordAfter.getElementsByTagName("subword");
						
						if (subwordsAfter.getLength() == 0) {
							example.setPosAfter(wordAfter.getAttribute("pos"));
						}
						else {
							Element subwordAfter = (Element) subwordsAfter.item(0);
							example.setPosAfter(subwordAfter.getAttribute("pos"));
						}
					}
				}
				else {
					Element subwordAfter = (Element) contextSubwords.item(id + 1);
					example.setPosAfter(subwordAfter.getAttribute("pos"));
				}
			}
		} 
		else if (pos.startsWith("n") || pos.startsWith("v") || pos.startsWith("a")) {
			example.addContextWords(token);
		} 
	}
	
}
