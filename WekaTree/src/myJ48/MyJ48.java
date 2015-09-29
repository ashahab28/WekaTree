/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myJ48;

import myID3.*;
import WekaInterface.Weka;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.filters.Filter;

/**
 *
 * @author ahmadshahab
 */
public class MyJ48 extends AbstractClassifier{
    
    /** Attribute of this class */
    
    public MyJ48[] nodes;
    
    // For splitting the tree
    public Attribute currentAttribute;
    
    // If leaf, then the value is class value
    public double classValue;
    
    // Class distribution (if leaf)
    public double[] classDistribution;
    
    // Attribute identity for the class of the node (if leaf)
    public Attribute classAttribute;
    
    //
    public MyJ48 predecessor;
    
    //a
    public double initAccuracy;
    
    public boolean visited;
        
    public boolean classify;
    
    public MyJ48(){
        visited = false;
    }
    
    public MyJ48(MyJ48 myJ48){
        visited = false;
        this.predecessor = myJ48;
    }
    
    /**
     * Build an Id3 classifier
     * @param instances dataset used for building the model
     * @throws Exception 
     */
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        
         if(instances == null){
                    //System.out.println("instances null");
                }
                else{
                    //System.out.println("instances ga null");
                }
        
        // Detecting the instance type, can Id3 handle the data?
        getCapabilities().testWithFail(instances);
        
        // Remove missing class
        Instances data = new Instances(instances);
        data.deleteWithMissingClass();
        
        // Build the id3
        buildTree(data);
        
        Weka weka = new Weka();
        String[] options_cl = {""};
        weka.setTraining("weather.nominal.arff");
        weka.setClassifier("weka.classifiers.trees.J48", options_cl);

        weka.runCV(false);
        
        initAccuracy = weka.getM_Evaluation().correct();
        
         if(data == null){
                    //System.out.println("data null");
                }
                else{
                    //System.out.println("data ga null");
                }
        
        pruneTree(data);
    }
    
    /**
     * Construct the tree using the given instance
     * Find the highest attribute value which best at dividing the data
     * @param data Instance
     */
    public void buildTree(Instances data) throws Exception{
        if(data.numInstances() > 0){
            // Lets find the highest Information Gain!
            // First compute each information gain attribute
            double IG[] = new double[data.numAttributes()];
            Enumeration enumAttribute = data.enumerateAttributes();
            while(enumAttribute.hasMoreElements()){
                Attribute attribute = (Attribute)enumAttribute.nextElement();
                IG[attribute.index()] = informationGain(data, attribute);
                // System.out.println(attribute.toString() + ": " + IG[attribute.index()]);
            }
            // Assign it as the tree attribute!
            currentAttribute = data.attribute(maxIndex(IG));
            //System.out.println(Arrays.toString(IG) + IG[currentAttribute.index()]);
            
            // IG = 0 then current node = leaf!
            if(Utils.eq(IG[currentAttribute.index()], 0)){
                // Set the class value as the highest frequency of the class
                currentAttribute = null;
                classDistribution = new double[data.numClasses()];
                Enumeration enumInstance = data.enumerateInstances();
                while(enumInstance.hasMoreElements()){
                    Instance temp = (Instance)enumInstance.nextElement();
                    classDistribution[(int) temp.classValue()]++;
                }
                Utils.normalize(classDistribution);
                classValue = Utils.maxIndex(classDistribution);
                classAttribute = data.classAttribute();
            }
            else{
                // Create another node from the current tree
                Instances[] splitData = splitDataByAttribute(data, currentAttribute);
                nodes = new MyJ48[currentAttribute.numValues()];
                
                for (int i = 0; i < currentAttribute.numValues(); i++) {
                    nodes[i] = new MyJ48(this);
                    nodes[i].buildTree(splitData[i]);
                }
            }
        }
        else{
            classAttribute = null;
            classValue = Utils.missingValue();
            classDistribution = new double[data.numClasses()];
        }
        
    }
    
    /**
     * Construct the tree using the given instance
     * Find the highest attribute value which best at dividing the data
     * @param data Instance
     */
    public void pruneTree2(Instances data) throws Exception{
        if (currentAttribute == null) {
            Attribute tempAttr = predecessor.currentAttribute;  
            predecessor.currentAttribute = null;
            // Set the class value as the highest frequency of the class
            classDistribution = new double[data.numClasses()];
            Enumeration enumInstance = data.enumerateInstances();
            while(enumInstance.hasMoreElements()){
                Instance temp = (Instance)enumInstance.nextElement();
                classDistribution[(int) temp.classValue()]++;
            }
            Utils.normalize(classDistribution);
            predecessor.classValue = Utils.maxIndex(classDistribution);
            predecessor.classAttribute = data.classAttribute();
            Weka weka = new Weka();
            weka.setTraining("weather.nominal.arff");
            String[] options_cl = {""};
            weka.setClassifier("myJ48.MyJ48", options_cl);
            
            weka.runCV(true);
            double currentAccuracy = weka.getM_Evaluation().correct();
            double maxFalseAccuracy = initAccuracy * 0.9;
            
            if(maxFalseAccuracy > currentAccuracy){
                predecessor.currentAttribute = tempAttr;
                visited = true;
            }
            else{
                visited = false;
            }
        }
        else if(visited){
        }
        else {
            for (int j = 0; j < currentAttribute.numValues(); j++) {
                if(nodes[j] == null){
                    //System.out.println("null nodes");
                }
                else{
                    //System.out.println("ga null");
                }
                nodes[j].pruneTree(data);
           }
        }
    }
    
    public MyJ48 pruneTree(Instances data) throws Exception{
        
    }
    
    /**
     * Count the information gain for selected attribute 
     * from the given dataset
     * @param data
     * @param attribute
     * @return 
     */
    private double informationGain(Instances data, Attribute attribute){
        
        /* Information Gain = Init entropy - After change entropy */
        
        double initEntropy = entropy(data);
        
        // Now we split the attribute first to count each entropy on different value
        Instances[] subSet = splitDataByAttribute(data, attribute);
        double[] entropy = new double[attribute.numValues()];
        
        // Count the entropy!
        for(int i = 0; i < attribute.numValues(); i++){
            if(subSet[i].numInstances() > 0) entropy[i] = entropy(subSet[i]);
            else entropy[i] = 0;
        }
        //System.out.println(attribute.toString() + " " + Arrays.toString(entropy) + "\n");
        
        double IG = initEntropy;
        
        for (int i = 0; i < attribute.numValues(); i++) {
            IG = IG - (entropy[i]*(double)subSet[i].numInstances()/data.numInstances());
        }
        
        return IG;
    } 
    
    /**
     * Find the entropy from a given dataset
     * @param data
     * @return 
     */
    private double entropy(Instances data){
    
        /*  Entropy = -(p1 log2 p1) -(p2 log2 p2).... */
        
        double numInstance = data.numInstances();
        double numClass = data.numClasses();
        double[] distribution = new double[data.numClasses()];
        
        Enumeration instance = data.enumerateInstances();
        while(instance.hasMoreElements()){
            Instance temp = (Instance) instance.nextElement();
            /* Count the p1, p2 */
            distribution[(int)temp.classValue()] ++;
        }
        

        /* Sum all the distribution */
        double sum = 0;
        for(int i = 0; i < numClass; i++){
            distribution[i] = distribution[i]/numInstance;
            if(distribution[i] > 0.0)
                distribution[i] *= Utils.log2(distribution[i]);
            // System.out.println(Arrays.toString(distribution));
            sum += distribution[i];
        }
            
        return -1 * sum;
    }
    
    /**
     * Create split of data based on the value of attribute
     * @param data
     * @param attribute
     * @return 
     */
    private Instances[] splitDataByAttribute(Instances data, Attribute attribute){
        
        // Init the object first
        Instances[] subSet = new Instances[attribute.numValues()];
        for (int i = 0; i < attribute.numValues(); i++) {
            subSet[i] = new Instances(data, data.numInstances());
        }
        
        // Split it!
        Enumeration instanceEnum = data.enumerateInstances();
        while(instanceEnum.hasMoreElements()){
            Instance instance = (Instance)instanceEnum.nextElement();
            subSet[(int)instance.value(attribute)].add(instance);    
        }
        
        // Compact the array of object by removing the empty array
        for (int i = 0; i < attribute.numValues(); i++) {
            subSet[i].compactify();
            // System.out.println(subSet[i]);
        }
        
        return subSet;
    }
    
    /**
     * Capability of id3 classifier
     * @return 
     */
    @Override
    public Capabilities getCapabilities(){
        Capabilities id3_capability = super.getCapabilities();
        id3_capability.disableAll();
        
        // Attribute type capability
        id3_capability.enable(Capability.NOMINAL_ATTRIBUTES);
        
        // Class capability
        id3_capability.enable(Capability.NOMINAL_CLASS);
        id3_capability.enable(Capability.MISSING_CLASS_VALUES);
        
        // Minimum number of instances allowed to be use
        id3_capability.setMinimumNumberInstances(0);
        
        return id3_capability;
    }
    
    private int maxIndex(double[] arr){        
        double max_val = 0;
        int max_index = 0;
        for(int i = 0; i < arr.length; i++){
            if(arr[i] == Double.NaN){
                arr[i] = -9.9;
            }
            
            if(max_val <= arr[i]){
                max_val = arr[i];
                max_index = i;
            }
        }
        return max_index;
    }
    
    /**
    * Classifies a given test instance using the decision tree.
    *
    * @param instance the instance to be classified
    * @return the classification
    * @throws NoSupportForMissingValuesException if instance has missing values
    */
    public double classifyInstance(Instance instance) throws NoSupportForMissingValuesException {
      if (instance.hasMissingValue()) {
        throw new NoSupportForMissingValuesException("Id3: no missing values, "
                                                     + "please.");
      }
      if (currentAttribute == null) {
        return classValue;
      } else {
        return nodes[(int) instance.value(currentAttribute)].
          classifyInstance(instance);
      }
    }

   /**
    * Computes class distribution for instance using decision tree.
    *
    * @param instance the instance for which distribution is to be computed
    * @return the class distribution for the given instance
    * @throws NoSupportForMissingValuesException if instance has missing values
    */
    public double[] distributionForInstance(Instance instance) throws NoSupportForMissingValuesException {
      if (instance.hasMissingValue()) {
        throw new NoSupportForMissingValuesException("Id3: no missing values, "
                                                     + "please.");
      }
      if (currentAttribute == null) {
        return classDistribution;
      } else { 
        return nodes[(int) instance.value(currentAttribute)].
          distributionForInstance(instance);
      }
    }
   
    /**
    * Prints the decision tree using the private toString method from below.
    *
    * @return a textual description of the classifier
    */
    @Override
    public String toString() {

      if ((classDistribution == null) && (nodes == null)) {
        return "Id3: No model built yet.";
      }
      return "Id3\n\n" + printTree(0);
    }
   
    public String printTree(int level){
        StringBuilder text = new StringBuilder();

         if (currentAttribute == null) {
           if (Utils.isMissingValue(classValue)) {
             text.append(": null");
           } else {
             text.append(": ").append(classAttribute.value((int) classValue));
           } 
         } else {
           for (int j = 0; j < currentAttribute.numValues(); j++) {
             text.append("\n");
             for (int i = 0; i < level; i++) {
               text.append("|  ");
             }
             text.append(currentAttribute.name()).append(" = ").append(currentAttribute.value(j));
             text.append(nodes[j].printTree(level + 1));
           }
         }
         return text.toString();
    }
    
    public void NumericToNominal(Instances numericSet) throws Exception {
        //TODO: cek atribut numerik
        double maxIG, curIG;
        int maxIdx;
        for(int i=0;i<numericSet.numAttributes();i++){
            //System.out.println(numericSet.attribute(i).name());
            Instances tempSet;
            if(numericSet.attribute(i).isNumeric()){
                //System.out.println("is numeric");
                maxIG = 0;
                maxIdx = 0;
                //Instances[] NominalizedSets = new Instances[numericSet.numInstances()];
                for(int ix=0;ix<numericSet.numInstances();ix++){
                    tempSet = NumericToNominalByThreshold(numericSet, i, numericSet.instance(ix).value(i));
                    //System.out.println(tempSet);
                    //System.out.println(tempSet.attribute(i));
                    //System.exit(1);
                    curIG = informationGain(tempSet, tempSet.attribute(i));
                    System.out.println("by value index: " + ix + " IG: " + curIG);
                    if(maxIG<curIG){
                        maxIG = curIG;
                        maxIdx = ix;
                    }
                }
                numericSet = NumericToNominalByThreshold(numericSet, i, numericSet.instance(maxIdx).value(i));
                System.out.println("Nominalized by attribute " + numericSet.attribute(i).name() + " :\n" + numericSet.toString());
                System.out.println("max index: "+ maxIdx);
                System.out.println("max IG: " + informationGain(numericSet, numericSet.attribute(i)));
            }
        }
    }
    
    public Instances NumericToNominalByThreshold(Instances numericSet, int idx_attribute, double threshold) throws Exception{
        double[] values;
        Instances NominalizedSet = new Instances(numericSet);
        //System.out.println("number of instances: " + NominalizedSet.numInstances());
        values = numericSet.attributeToDoubleArray(idx_attribute);
        List<String> nominalValue = new ArrayList<String>();
        nominalValue.add("low");
        nominalValue.add("high");
        Attribute nominalAttrib = new Attribute(numericSet.attribute(idx_attribute).name() + "_nominal",nominalValue);
        NominalizedSet.insertAttributeAt(nominalAttrib,idx_attribute);
        for(int i=0;i<values.length;i++){
            if(values[i]<=threshold){
                NominalizedSet.instance(i).setValue(idx_attribute, "low");
            }
            else{
                NominalizedSet.instance(i).setValue(idx_attribute, "high");
            }
        }
        String[] options = {"-R", String.valueOf(idx_attribute+2)};
        Filter remove = (Filter) Class.forName("weka.filters.unsupervised.attribute.Remove").newInstance();
        ((OptionHandler) remove).setOptions(options);
        remove.setInputFormat(NominalizedSet);
        NominalizedSet = Filter.useFilter(NominalizedSet, remove);
        
        return NominalizedSet;
    }

   public static void main(String[] args) throws IOException, Exception{
        Weka wk = new Weka();
        wk.setTraining("iris.2D.arff");
        MyJ48 j48 = new MyJ48();
        j48.NumericToNominal(wk.getM_Training());
   }
}
