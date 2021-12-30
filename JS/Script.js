	//Data
		//[(training or unknown), (dataset), (input or output), (data)] (3 in output means unknown)
		var data =
			[
				[
					[[1,    0.5], [0, 1]],
					[[1.5, 0.75], [0, 1]],
					[[1.5,    1], [0, 1]],
					[[1.75,   1], [0, 1]],
					
					[[2,      1], [1, 0]],
					[[2.5, 1.25], [1, 0]],
          [[3,      2], [1, 0]],
          [[2.75,   2], [1, 0]],
				],
				
				[
          [[3,   2.25], [3, 3]],
          [[1.5,    1], [3, 3]],
				]
			];
		
		
	//Helpers (called here due to memory issues)
		var i = 0, k = 0, t = 0, h = 0, q = 0, LayerDescriber = 0, NumDescriber = 0, change = 0, TargetVariable = 0, Slope = 0;
	
  
	//Defining Sigmoid
		function sig(x) {return (1 / (1 + Math.pow(2.718, -x))); };
  
  
	//Asking for the size of the network
		var NumOfHidLayers = prompt("The number of hidden layers \n (min of 1):");
		var EveryHidLayerSize = prompt("The size of each hidden layer: \n (min of " + Math.max(data[0][0][0].length, data[0][0][1].length) + ")");
		var NumOfTrainingLoop = prompt("The number training loops:");
		var TrainingRate = prompt("The training rate (slope * rate): \n (exp: slope * 0.5)");
	
  
	//Weight config (Bias included)
		var NumOfInputData = data[0][0][0].length;
		var NumOfOutputs = data[0][0][1].length;
		var NumOfWeights = ( NumOfInputData * EveryHidLayerSize ) + ( (NumOfHidLayers - 1) * Math.pow(EveryHidLayerSize, 2) ) + (EveryHidLayerSize * NumOfOutputs);
		var w = [];
		 for(count = 0; count <= NumOfHidLayers; count++) {
    	 w[count] = [];
     };
     k = 0, t = 0;
     for(count = 0; count < Math.pow(EveryHidLayerSize, 2) * (NumOfHidLayers + 1); count++) {
			 w[k][t] = Math.random() * 10;
			 t++;
			 if(t == Math.pow(EveryHidLayerSize, 2)) {
				 k++;
				 t = 0;
			 };
		 };
		var bias = [];
		 for(count = 0; count <= NumOfHidLayers; count++) {
			 bias[count] = Math.random() * 10;
		 };
    
    
    
	//Neural Net Config******************************************************************************************************
    var node = [];
     for(count = 0; count < NumOfHidLayers; count++) {
    	 node[count] = [];
     };
		var NN = [];
	  //[(type: output or training), (i: Selected Data)]
		function NN_Construction(type, i) {
     //Reseting all nodes
      k = 0, t = 0;
      for(count = 0; count < (NumOfHidLayers * EveryHidLayerSize); count++) {
				node[k][t] = 0;
				t++;
				if(t == EveryHidLayerSize) {
					k++;
					t = 0;
				};
			};
      for(count = 0; count < NumOfOutputs; count++) {
    	  NN[count] = 0;
      };
      
      
		 //First hidden layer
			k = 0, t = 0;
			for(count = 0; count < (NumOfInputData * EveryHidLayerSize); count++) {
				node[0][k] += data[type][i][0][t] * w[0][count];
				t++;
				if(t == NumOfInputData) {
					node[0][k] += bias[0];
					k++;
					t = 0;
				};
			};
      
      
		 //configuring the hidden layers
		  k = 0, t = 0, h = 0, q = 0;
			for(count = 0; count < (NumOfHidLayers - 1) * Math.pow(EveryHidLayerSize, 2); count++) {
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
		    k = 0, t = 0, q = 0;
		    for(count = 0; count < (NumOfOutputs * EveryHidLayerSize); count++) {
		    	NN[k] += node[NumOfHidLayers - 1][t] * w[NumOfHidLayers][q];
		    	t++;
		    	q++;
		    	if(t == EveryHidLayerSize) {
		    		NN[k] += bias[NumOfHidLayers];
		    		k++;
		    		t = 0;
		    	};
		    };
		    
		    	
		};
	//Neural Net Config "END"*************************************************************************************************
		
    
		
	//Training****************************************************************************************************************
  	var TempW = [];
     for(count = 0; count <= NumOfHidLayers; count++) {
       TempW[count] = [];
     };
    var TempB = [];
  	var TempNN = [];
		change = 0;
    q = 0;
	 	for(counter = 0; counter < NumOfTrainingLoop * (NumOfWeights + bias.length); counter++) {
	 		//selecting a random input data
	    	i = Math.floor(Math.random() * data[0].length);
	    	
	    //Catching our current NN result
	    	NN_Construction(0, i);
	    	
	    //Copying the NN
	    	for(count = 0; count < NumOfOutputs; count++) {
        	TempNN[count] = NN[count];
        };
	    	
	    //Finding the Slope------------------------------------
	    	//Copying weights and biases
	    		k = 0, t = 0;
	    		for(count = 0; count < Math.pow(EveryHidLayerSize, 2) * (NumOfHidLayers + 1); count++) {
	    			TempW[k][t] = w[k][t];
	    			t++;
	    			if(t == Math.pow(EveryHidLayerSize, 2)) {
	    				k++;
	    				t = 0;
	    			};
	    		};
			
	    		for(count = 0; count <= NumOfHidLayers; count++) {
	    			TempB[count] = bias[count];
	    		};
	    		
	    	//Derivative for Weights
	    		if(q < NumOfWeights) {
   					TargetVariable = w[LayerDescriber][NumDescriber];
	  		 		k = 0, t = 0;
	    			for(count = 0; count < Math.pow(EveryHidLayerSize, 2); count++) {
	   					w[LayerDescriber][count] = 0;
	    			};
	    			w[LayerDescriber][NumDescriber] = TargetVariable;
	    			for(count = LayerDescriber; count <= NumOfHidLayers; count++) {
            	bias[count] = 0;
            };
	    		}
		    
	    	//Derivative for Bias
	    		else if(q < (NumOfWeights + bias.length) &&  q >= NumOfWeights) {
	    			TargetVariable = bias[LayerDescriber];
	    			k = 0, t = 0;
	    			for(count = 0; count < Math.pow(EveryHidLayerSize, 2); count++) {
	    				w[0][count] = 0;
	    			};
	    			for(count = 0; count <= NumOfHidLayers; count++) {
            	bias[count] = 0;
            };
	    			bias[LayerDescriber] = TargetVariable;
	    		};
          
          
	    		
	    		NN_Construction(0, i);
	    		h =  Math.floor(Math.random() * NumOfOutputs);
          //for when we are taking drivitive of a weight in the last layer
	    		 if(LayerDescriber == NumOfHidLayers && q < NumOfWeights) {
	    			 h = Math.floor(NumDescriber / EveryHidLayerSize);
	    		 };
	    		Slope = (2 * NN[h] * sig(TempNN[h]) * (1 - sig(TempNN[h])) * (sig(TempNN[h]) - data[0][i][1][h]));
	    		k = 0, t = 0;
	    		for(count = 0; count < Math.pow(EveryHidLayerSize, 2) * (NumOfHidLayers + 1); count++) {
	    			w[k][t] = TempW[k][t];
	    			t++;
	    			if(t == Math.pow(EveryHidLayerSize, 2)) {
	    				k++;
	    				t = 0;
	    			};
	    		};
	    		for(count = 0; count <= NumOfHidLayers; count++) {
          	bias[count] = TempB[count];
          };
		    
	    	//Finding the Slope "END"------------------------------
	    		
	    		
	    	//Improving the weights -------------------------------
	    			change = Slope * TrainingRate;
	    			if(q < NumOfWeights) {
	    				w[LayerDescriber][NumDescriber] -= change;
	    				NumDescriber++;
              if(NumDescriber == Math.pow(EveryHidLayerSize, 2)) {
	    					LayerDescriber++;
	    					NumDescriber = 0;
	    				} else if(LayerDescriber == 0 && NumDescriber == NumOfInputData * EveryHidLayerSize) {
	    					LayerDescriber++;
	    					NumDescriber = 0;
	    				} else if(LayerDescriber == NumOfHidLayers && NumDescriber == NumOfOutputs * EveryHidLayerSize) {
	    					LayerDescriber = 0;
	    					NumDescriber = 0;
	    				};
	    			}
            
	    			else if(q < (NumOfWeights + bias.length) &&  q >= NumOfWeights) {
	    				bias[LayerDescriber] -= change;
	    				LayerDescriber++;
	    				if(q == NumOfWeights + bias.length - 1) {
	    					LayerDescriber = 0;
	    					q = -1;
	    				};
	    			};
            q++;
	    	//Improving the weights "END"------------------------------
	    			
	    	
	    	
		};
	//Training "END"**********************************************************************************************************
		
		
		
	//Request for answer
		var cate = [];
		if(true) {
			cate[0] = "Red Flower";
			cate[1] = "Blue Flower";
		} else {
			for(count = 0; count < NumOfOutputs ; count++) {
   			cate[count] = "Category Number " + (count + 1);
     	};
		};
    
		for(counter = 1; counter <= data[1].length ; counter++) {
			alert("For dataset number " + counter + ":");
			NN_Construction(1, counter - 1);
			for(count = 0; count < NumOfOutputs ; count++) {
				Shorter = Math.floor(sig(NN[count]) * 100000) / 1000;
        if(sig(NN[count]) >= 0.6) {
					alert("          I'm " + Shorter + "% sure it's a " + cate[count]);
				}
        else if(sig(NN[count]) <= 0.4) {
					alert("          I'm " + (100 - Shorter) + "% sure it's |NOT| a " + cate[count]);
				}
        else if(sig(NN[count]) > 0.4 && sig(NN[count]) < 0.6) {
					alert("          I'm uncertain (P = " + Shorter/100 + ")");
				}
      };
    };
    