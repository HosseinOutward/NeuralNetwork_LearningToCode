//training set. [input1, input2, output(s)]
var data = [
  [1,   0.5, [0]],
  [1.5, 1,   [0]],
  [1.5, 1.5, [0]],
  [2,   1.5, [0]], 

  [3,   2,   [1]],
  [3,   1.5, [1]],
  [3,   1,   [1]],
  [3.5, 1,   [1]],
];

//unknown type (data we want to find)
var dataU = [
  [4,   2, "it should be 1"]
];

//asking for the size of the network
var NumOfHidLayers = 1//prompt("The number hidden layers:");
if(NumOfHidLayers == 0) {EveryHidLayerSize = 0;} else {
	var EveryHidLayerSize = 2//prompt("The size of each hidden layer:");
};
var NumOfTrainingLoop = 50000//prompt("The number training loops:");
var TrainingRate = 0.1//prompt("The training rate (slope * X)\n  (exp: slope * 0.5):");


//Weights config (bias included)
var NumOfInputLayers = data[0].length - 1;
var NumOfOutputs = data[0][data[0].length - 1].length
var NumOfWeights = (NumOfInputLayers * EveryHidLayerSize) + ( (NumOfHidLayers - 1) * Math.pow(EveryHidLayerSize, 2) ) + (EveryHidLayerSize * NumOfOutputs);
  if(NumOfHidLayers == 0) {NumOfWeights = NumOfInputLayers * NumOfOutputs};

var bias = [0]
for(count = 0; count < (NumOfHidLayers + 1); count++) {
  bias[count] = Math.random()
};

var k = 0;
var t = 0;
var w = []
for(count = 0; count <= NumOfHidLayers; count++) {w[count] = [];};
  if(EveryHidLayerSize == 0) {w[0] = [];}
for(count = 0; count < NumOfWeights; count++) {
  w[k][t] = Math.random();
  t++;
  if(t == Math.pow(EveryHidLayerSize, 2)) {
    k++;
    t = 0;
  };
};


//Neural net Result (i: data selector)
var NN = [];
function NN_Construction(i) {
	//Neural nets config **********************************************************
   
 	  //reseting all nodes to 0  
    var node = [[]];
    NN = [];
    k = 0;
    t = 0;
    for(count = 0; count < NumOfOutputs; count++) {NN[count] = 0};
    for(count = 0; count < NumOfHidLayers; count++) {node[count] = [];};
		for(count = 0; count < (NumOfHidLayers * EveryHidLayerSize); count++) {
			node[k][t] = 0;
      t++;
      if(t == EveryHidLayerSize) {
    	  k++;
    	  t = 0;
      };
		};
    
	  //First hidden layer
		k = 0
    t = 0
		for(count = 0; count < (NumOfInputLayers * EveryHidLayerSize); count++) {
      node[0][k] += data[i][t] * w[0][count];
      t++;
      if(t == NumOfInputLayers) {
     	 node[0][k] += bias[0];
     	 k++;
     	 t = 0;
      };
		};

	  //Rest of the hidden layers
    k = 1;
    t = 0;
    var h = 0;
    var q = 0;
    var b = 1;
		for(count = 0; count < ( (NumOfHidLayers - 1) * Math.pow(EveryHidLayerSize, 2) ); count++) {
      node[k][t] += node[k - 1][h] * w[k][q];
      h++;
      q++
      if(h == EveryHidLayerSize) {
        node[k][t] += bias[b];
     	  h = 0;
        t++
        if(t == EveryHidLayerSize) {
       	 b++;
       	 k++;
         q = 0
       	 t = 0; 
        }
      };
    };

		//Output layer 
    k = 0;
    t = 0;
    b = NumOfHidLayers
    q = 0
    for(count = 0; count < (NumOfOutputs * EveryHidLayerSize); count++) {
    	NN[k] += node[NumOfHidLayers - 1][t] * w[NumOfHidLayers][q];
      t++;
      q++;
      if(t == EveryHidLayerSize) {
     	 NN[k] += bias[b];
       b++
     	 k++;
     	 t = 0;
      };
    };
    //**In case we had no hidden layer
    if(NumOfHidLayers == 0) {
    	k = 0
    	t = 0
      b = 0
      q = 0
    	for(count = 0; count < (NumOfInputLayers * NumOfOutputs); count++) {
      	NN[k] += data[i][t] * w[0][q];
      	t++;
        q++;
      	if(t == NumOfInputLayers) {
     	 		NN[k] += bias[b];
     	 		k++;
     	 		t = 0;
      	};
			};
    };
    //***************************************************************
};




//Defining sigmoid
function Sig(x) {return (1 / (1 + Math.pow(2.718, -x))); };




//Defining Derivative of sigmoid with respect to NN
function dSig_dNN(i, NN, counter) {
	return( 2 * Sig(NN[counter]) * (1 - Sig(NN[counter])) * (Sig(NN[counter]) - data[i][NumOfInputLayers][counter]) * NN[counter] );
}


//Slope and stuff (i: data selector, r: is it a weight or bias, k,t,b: describing variuable)
function SlopeOfCost(i, r, k, t, b) {
	//Cost Slope config **********************************************************
  
 	  //reseting all nodes to 0  
    var node = [[]];
    var NN = [];
    k = 0;
    t = 0;
    for(count = 0; count < NumOfOutputs; count++) {NN[count] = 0};
    for(count = 0; count < NumOfHidLayers; count++) {node[count] = [];};
		for(count = 0; count < (NumOfHidLayers * EveryHidLayerSize); count++) {
			node[k][t] = 0;
      t++;
      if(t == EveryHidLayerSize) {
    	  k++;
    	  t = 0;
      };
		};
    
    //Derivative for Weights
    if(r == 1) {
    	var TemporaryValue = w[k][t]
      //reseting all other weights to 0 in the layer
			let w = []
			for(count = 0; count <= NumOfHidLayers; count++) {w[count] = [];};
  			if(EveryHidLayerSize == 0) {w[0] = [];}
			for(count = 0; count < Math.pow(EveryHidLayerSize, 2); count++) {
  			w[k][count] = 0;
			};
      w[k][t] = TemporaryValue
    };
    
    
    if(r == 0) {
    	var TemporaryValue = bias[b]
			let w = []
			for(count = 0; count < Math.pow(EveryHidLayerSize, 2); count++) {
  			w[0][count] = 0;
			};
      
      let bias = []
			for(count = 0; count < (NumOfHidLayers + 1); count++) {
  			bias[count] = 0
			};
      bias[b] = TemporaryValue
    };
    
    NN_Construction(i)
    var Slope;
    for(count = 0; count < NumOfOutputs; count++) { Slope += dSig_dNN(i, NN, count) };
    
    
	return(Slope)
  //***************************************************************
};


function Training(NumOfTrainingLoop, TrainingRate) {
//training **************************************************
  var i;
  var k = -1;
  var t = -1;
  var b = -1;
  var q = -1;
  for(count = 0; count < NumOfTrainingLoop; count++) {
    //Find a randome input data
    i = Math.floor(Math.random() * NumOfInputLayers)
    q++
  	if(q < NumOfWeights) {
    	//w[k][t]
      t++;
  		if(t == (Math.pow(EveryHidLayerSize, 2) - 1) ) {
    		k++;
    		t = -1;
      };
      
      var change = SlopeOfCost(i, 1, k, t, b) * TrainingRate
      w[k][t] -= change

    
  	} else {
    
    	if(q < NumOfWeights + bias.length) {
    		//bias[b]
      	b++;
      	var change = SlopeOfCost(i, 1, k, t, b) * TrainingRate
        bias[b] -= change
      	
      } else {
      	k = -1
  			t = -1
        b = -1;
  	  	q = -1; 
      };
    };
  };
//***************************************************************
};

Training(NumOfTrainingLoop, TrainingRate)



//request for answer

alert(Sig(NN[0]))

