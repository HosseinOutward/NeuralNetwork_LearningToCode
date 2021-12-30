public class NN_Basic {
	
	
	//Helpers (called here due to memory issues)
	private static int i, k, t, h, q, LayerDescriber, NumDescriber = 0;
	private static double change, TargetVariable, Slope = 0;
	
	//Defining Sigmoid
	private static double sig(double x) {return (1 / (1 + Math.pow(2.718, -x))); };
	
	
	
	public static void main(String[] args) {
		
	//Defining prompt and line break
		java.util.Scanner prompt = new java.util.Scanner(System.in);
		String LineBreak = System.getProperty("line.separator");
		
		
	//Data
		//[(training or unknown), (dataset), (input or output), (data)] (3 in output means unknown)
		double data[][][][] =
			{
				{
					{{1,   0.5}, {0, 1}},
					{{1.5, 1},   {0, 1}},
					{{1.5, 1.5}, {0, 1}},
					{{2,   1.5}, {0, 1}},
					
					{{3,   2},   {1, 0}},
					{{3,   1.5}, {1, 0}},
					{{3,   1.75},{1, 0}},
					{{3.5, 1},   {1, 0}}
				},
				
				{
					{{3,    2},    {3}},
					{{1.75, 0.75}, {3}},
					{{2.5,  2},    {3}}
				}
			};
		
		
		
	//Asking for the size of the network
		System.out.println("The number of hidden layers (min of 1):");
		int NumOfHidLayers = 3;//prompt.nextInt();
		
		System.out.println("The size of each hidden layer:" + LineBreak + 
				"make sure the number is bigger than " + Math.max(data[0][0][0].length, data[0][0][1].length) );      
		int EveryHidLayerSize = 3;//prompt.nextInt();

		System.out.println("The number training loops:");
		int NumOfTrainingLoop = 100;//prompt.nextInt();

		System.out.println("The training rate (slope * rate)" + LineBreak + "(exp: slope * 0.5):");
		double TrainingRate = 0.1;//prompt.nextDouble();
		
		prompt.close();

		
		
	//Weight config (Bias included)
		int NumOfInputData = data[0][0][0].length;
		int NumOfOutputs = data[0][0][1].length;
		int NumOfWeights = (int) (( NumOfInputData * EveryHidLayerSize )
				+ ( (NumOfHidLayers - 1) * Math.pow(EveryHidLayerSize, 2) ) + (EveryHidLayerSize * NumOfOutputs));
			if(NumOfHidLayers == 0) { NumOfWeights = NumOfInputData * NumOfOutputs; };

		double bias[] = new double[(int) NumOfHidLayers + 1];
		for(int count = 0; count <= NumOfHidLayers; count++) {
			bias[count] = Math.random();
		};

		double w[][] = new double[(int) NumOfHidLayers + 1][(int) Math.pow(EveryHidLayerSize, 2)];
		for(int count = 0; count < Math.pow(EveryHidLayerSize, 2) * (NumOfHidLayers + 1); count++) {
			w[k][t] = Math.random();
			t++;
			if(t == Math.pow(EveryHidLayerSize, 2)) {
				k++;
				t = 0;
			};
		};
		
		
	//Neural Net Config***************************************************
		double NN[] = new double[NumOfOutputs];
		double node[][] = new double[NumOfHidLayers][EveryHidLayerSize];
		//[(type: output or training), (i: Selected Data)]
		java.util.function.BiConsumer<Integer, Integer> NN_Construction = (type, i) -> {
		//First hidden layer
			k = 0;
			t = 0;
			for(int count = 0; count < (NumOfInputData * EveryHidLayerSize); count++) {
				node[0][k] += data[type][i][0][t] * w[0][count];
				t++;
				if(t == NumOfInputData) {
					node[0][k] += bias[0];
					k++;
					t = 0;
				};
			};
			
			
		//Rest of the hidden layers
		    k = 0;
		    t = 0;
		    h = 0;
		    q = 0;
			for(int count = 0; count < (NumOfHidLayers - 1) * Math.pow(EveryHidLayerSize, 2); count++) {
				node[k + 1][t] += node[k][h] * w[k + 1][q];
				h++;
				q++;
				if(h == EveryHidLayerSize) {
					node[k + 1][t] += bias[k + 1];
					h = 0;
					t++;
					if(t == EveryHidLayerSize) {
		       	 		k++;
		       	 		q = 0;
		       	 		t = 0; 
					};
				};
		    };
		    
		    
		//Output layer 
		    k = 0;
		    t = 0;
		    q = 0;
		    for(int count = 0; count < (NumOfOutputs * EveryHidLayerSize); count++) {
		    	NN[q] += node[NumOfHidLayers - 1][k] * w[NumOfHidLayers][t];
		    	t++;
		    	k++;
		    	if(t == EveryHidLayerSize) {
		    		NN[q] += bias[NumOfHidLayers];
		    		q++;
		    		k = 0;
		    	};
		    };
		    
		    	
		};
	//Neural Net Config "END"**********************************************
		
		
		
	//Training*************************************************************
		double TempB[] = new double[(int) NumOfHidLayers + 1];
		double TempW[][] = new double[(int) NumOfHidLayers + 1][(int) Math.pow(EveryHidLayerSize, 2)];
		double TempNN[] = new double[NumOfOutputs];
		change = 0;
	  	q = 0;
	  	for(int counter = 0; counter < NumOfTrainingLoop * (NumOfWeights + bias.length); counter++) {
	  		//selecting a random input data
	    	i = (int) Math.floor(Math.random() * data[0].length);
	    	
	    	//The NN result
	    	NN_Construction.accept(0, i);
	    									System.out.println(Slope);
	    	//start
	    	for(int count = 0; count < NumOfOutputs; count++) {TempNN[count] = NN[count];};
	    	
	    	//Finding the Slope------------------------------------
	    		//Set temporary weights and biases
	    		k = 0;
	    		t = 0;
	    		for(int count = 0; count < Math.pow(EveryHidLayerSize, 2) * (NumOfHidLayers + 1); count++) {
	    			TempW[k][t] = w[k][t];
	    			t++;
	    			if(t == Math.pow(EveryHidLayerSize, 2)) {
	    				k++;
	    				t = 0;
	    			};
	    		};
			
	    		for(int count = 0; count <= NumOfHidLayers; count++) {
	    			TempB[count] = bias[count];
	    		};
		    
		    
	    		//.......................
	    		
	    			//Derivative for Weights
	    			if(q < NumOfWeights) {
	    				
	    				TargetVariable = w[LayerDescriber][NumDescriber];
	    			
	    				k = 0;
	    				t = 0;
	    				for(int count = 0; count < Math.pow(EveryHidLayerSize, 2); count++) {
	    					w[LayerDescriber][count] = 0;
	    				};
	    			
	    				w[LayerDescriber][NumDescriber] = TargetVariable;
		        
	    				for(int count = LayerDescriber; count <= NumOfHidLayers; count++) {bias[count] = 0;};
	    			};
		    
		    
	    			//Derivative for Bias
	    			if(q < (NumOfWeights + bias.length) &&  q >= NumOfWeights) {
	    			
	    				TargetVariable = bias[LayerDescriber];
	    			
	    				k = 0;
	    				t = 0;
	    				for(int count = 0; count < Math.pow(EveryHidLayerSize, 2); count++) {
	    					w[0][count] = 0;
	    				};
		        
	    				for(int count = 0; count <= NumOfHidLayers; count++) {bias[count] = 0;};
		        
	    				bias[LayerDescriber] = TargetVariable;
	    			};
	    		//"END"..................
		    
	    			
	    		NN_Construction.accept(0, i);
	    		h =  (int) Math.floor(Math.random() * NumOfOutputs);
	    		if(LayerDescriber == NumOfHidLayers && q < NumOfWeights) {
	    			h = (int) Math.floor(NumDescriber / EveryHidLayerSize);
	    		};
	    		Slope = (2 * NN[h] * sig(TempNN[h]) * (1 - sig(TempNN[h])) * (sig(TempNN[h]) - data[0][i][1][h]));
		    
	    		k = 0;
	    		t = 0;
	    		for(int count = 0; count < Math.pow(EveryHidLayerSize, 2) * (NumOfHidLayers + 1); count++) {
	    			w[k][t] = TempW[k][t];
	    			t++;
	    			if(t == Math.pow(EveryHidLayerSize, 2)) {
	    				k++;
	    				t = 0;
	    			};
	    		};
		        
	    		for(int count = 0; count <= NumOfHidLayers; count++) {bias[count] = TempB[count];};
		    
	    	//Finding the Slope "END"------------------------------
	    		
	    		
	    		//.......................
	    			change = Slope * TrainingRate;
	    			if(q < NumOfWeights) {
	    				w[LayerDescriber][NumDescriber] -= change;
	    				
	    				NumDescriber++;
	    				if(NumDescriber == NumOfOutputs * EveryHidLayerSize && LayerDescriber == NumOfHidLayers) {
	    					LayerDescriber = 0;
	    					NumDescriber = 0;
	    				};
	    				if(NumDescriber == NumOfInputData * EveryHidLayerSize && LayerDescriber == 0) {
	    					LayerDescriber++;
	    					NumDescriber = 0;
	    				};
	    				if(NumDescriber == Math.pow(EveryHidLayerSize, 2)) {
	    					LayerDescriber++;
	    					NumDescriber = 0;
	    				};
	    			};
			    
			    
	    			if(q < (NumOfWeights + bias.length) &&  q >= NumOfWeights) {
	    				bias[LayerDescriber] -= change;
	    				
	    				LayerDescriber++;
	    				if(q == NumOfWeights + bias.length - 1) {
	    					LayerDescriber = 0;
	    					q = -1;
	    				};
	    			};
	    		//"END"..................
	    			
	    	q++;
	    	
		};
	//Training "END"*******************************************************
		
		
		
	//Request for answer
		String cate[] = new String[NumOfOutputs];
		if(true) {
			cate[0] = "Red Flower";
			cate[1] = "Blue Flower";
		} else {
			for(int count = 0; count < NumOfOutputs ; count++) {cate[count] = "Category Number " + count + 1;};
		};
		
		
		double Shorter;
		for(int counter = 1; counter <= data[1].length ; counter++) {
			
			System.out.println("For dataset number " + counter + ":");
			
			NN_Construction.accept(1, counter - 1);
			
			for(int count = 0; count < NumOfOutputs ; count++) {
				
				Shorter = Math.floor(sig(NN[count]) * 100000) / 1000;
				
				if(sig(NN[count]) >= 0.6) {
					System.out.println("          I'm " + Shorter + "% sure it's a " + cate[count]);
				};
				
				if(sig(NN[count]) <= 0.4) {
					System.out.println("          I'm " + (100 - Shorter) + "% sure it's |NOT| a " + cate[count]);
				};
				
				if(sig(NN[count]) > 0.4 && sig(NN[count]) < 0.6) {
					System.out.println("          I'm unsure if it's a " + cate[count] + " (P = " + Shorter + "%)");
				};
			};	
		};
		
		
		
	}

}
