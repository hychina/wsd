package wsd.beans;

import java.util.ArrayList;
import java.util.List;

public class Example {
	private int id;
	private String posBefore = null;
	private String posAfter = null;
	private List<String> contextWords = new ArrayList<String>();
	
	public String getPosBefore() {
		return posBefore;
	}
	
	public void setPosBefore(String posBefore) {
		this.posBefore = posBefore;
	}
	
	public String getPosAfter() {
		return posAfter;
	}
	
	public void setPosAfter(String posAfter) {
		this.posAfter = posAfter;
	}

	public List<String> getContextWords() {
		return contextWords;
	}
	
	public void addContextWords(String word) {
		contextWords.add(word);
	}

	public int getId() {
		return id;
	}

	public void setId(int id) {
		this.id = id;
	}
}
