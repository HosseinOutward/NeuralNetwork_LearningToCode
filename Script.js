//training set. [training or unknown, input1, input2, output(s)]
var data = [
	[
  	[1,   0.5, [0]],
  	[1.5, 1,   [0]],
  	[1.5, 1.5, [0]],
  	[2,   1.5, [0]], 

  	[3,   2,   [1]],
  	[3,   1.5, [1]],
  	[3,   1,   [1]],
  	[3.5, 1,   [1]]
  ],

  [
  	[1,   0.75, "it should be 1"]
  ]
];

//Helpers (called here due to memory issues)
var k = 0;
var t = 0;
var h = 0;
var q = 0;
var b = 0;

//asking for the size of the network
var NumOfHidLayers = prompt("The number hidden layers:");
if(NumOfHidLayers == 0) {EveryHidLayerSize = 0;} else {
	var EveryHidLayerSize = prompt("The size of each hidden layer:");
};
var NumOfTrainingLoop = prompt("The number training loops:");
var TrainingRate = prompt("The training rate (slope * X)\n  (exp: slope * 0.5):");


//Weights config (bias included)
var NumOfInputLayers = data[0][0].length - 1;
var NumOfOutputs = data[0][0][data[0][0].length - 1].length
var NumOfWeights = (NumOfInputLayers * EveryHidLayerSize) + ( (NumOfHidLayers - 1) * Math.pow(EveryHidLayerSize, 2) ) + (EveryHidLayerSize * NumOfOutputs);
  if(NumOfHidLayers == 0) {NumOfWeights = NumOfInputLayers * NumOfOutputs};

var bias = [0]
for(count = 0; count <= NumOfHidLayers; count++) {
  bias[count] = Math.random()
};

var w = []
k = 0;
t = 0;
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

//Defining sigmoid
function Sig(x) {return (1 / (1 + Math.pow(2.718, -x))); };


//Neural net Result (i: data selector, type: training or unknown)
var NN = [];
function NN_Construction(type, i) {
	//Neural nets config **********************************************************
   
 	  //reseting all nodes to 0  
    var node = [[]];
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
      node[0][k] += data[type][i][t] * w[0][count];
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
    h = 0;
    q = 0;
    b = 1;
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
      	NN[k] += data[type][i][t] * w[0][q];
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


//Defining Derivative of sigmoid with respect to a variable in NN (I know the 2 is unnecessary)
function dSig_dNN(i, NN, counter) {
	return( 2 * NN * Sig(NN) * (1 - Sig(NN)) * (Sig(NN) - data[0][i][NumOfInputLayers][counter]) );
};

//Slope and stuff (i: data selector, type: is it a weight or bias, describers: describing variuable)
function SlopeOfCost(type, i, LayerDescriber, NumDescriber) {
	//Cost Slope config **********************************************************
		var TempW = []
    var TempB = []
		k = 0;
		t = 0;
		for(count = 0; count <= NumOfHidLayers; count++) {
    TempW[count] = [];
    TempB[count] = bias[count];
    };
  		if(EveryHidLayerSize == 0) {TempW[0] = [];}
		for(count = 0; count < NumOfWeights; count++) {
  		TempW[k][t] = w[k][t];
  		t++;
  			if(t == Math.pow(EveryHidLayerSize, 2)) {
    			k++;
    			t = 0;
  			};
			};
    
    
    //Derivative for Weights
    	if(type == 0) {
    		var TargetVariable = w[LayerDescriber][NumDescriber]
				k = 0;
				t = 0;
      	for(count = 0; count < Math.pow(EveryHidLayerSize, 2); count++) {
  				w[LayerDescriber][count] = 0;
				};
      	w[LayerDescriber][NumDescriber] = TargetVariable
        
        for(count = LayerDescriber; count <= NumOfHidLayers; count++) {bias[count] = 0;};
    };
    //...................
    
    //Derivative for Bias
    	if(type == 1) {
    		var TargetVariable = bias[LayerDescriber]
				k = 0;
				t = 0;
      	for(count = 0; count < Math.pow(EveryHidLayerSize, 2); count++) {
  				w[0][count] = 0;
				};
        
        for(count = 0; count <= NumOfHidLayers; count++) {bias[count] = 0;};
        
      	bias[LayerDescriber] = TargetVariable
    };
    //..................
    
        
    NN_Construction(0, i)
    var Slope = 0;
    for(count = 0; count < NumOfOutputs; count++) { Slope += dSig_dNN(i, NN, count) };
    
     k = 0;
     t = 0;
     for(count = 0; count < NumOfWeights; count++) {
			 w[k][t] = TempW[k][t];
  		 t++;
  		 if(t == Math.pow(EveryHidLayerSize, 2)) {
    		 k++;
    		 t = 0;
  		 };
		 };
        
        for(count = 0; count <= NumOfHidLayers; count++) {bias[count] = TempB[count];};
    
    
	return(Slope)
  //***************************************************************
};


function Training(NumOfTrainingLoop, TrainingRate) {
//training **************************************************
	//SlopeOfCost(i, type, LayerDescriber, NumDescriber)
  var i;
  k = 0;
  t = 0;
  b = 0;
  q = 0;
  var change;
  //SlopeOfCost(type, i, LayerDescriber, NumDescriber)
  for(count = 0; count < NumOfTrainingLoop; count++) {
    //selecting a randome input data
    i = Math.floor(Math.random() * NumOfInputLayers)
		switch (true) {
  		case (q < NumOfWeights): 
    		//w[k][t]
      	change = SlopeOfCost(0, i, k, t) * TrainingRate
      	w[k][t] -= change
      	t++;
  			if(t == Math.pow(EveryHidLayerSize, 2) ) {
    			k++;
    			t = 0;
      	};
  			break;
      
  		case (q < (NumOfWeights + bias.length) &&  q >= NumOfWeights):
				//bias[b]
      	change = SlopeOfCost(1, i, b, 0) * TrainingRate;
      	bias[b] -= change;
      	b++;
      	break;
      
  		case (q == NumOfWeights + bias.length):
    		k = 0;
  			t = 0;
  			b = 0;
  			q = 0;
    		break;
		};
    q++
  };
//***************************************************************
};

Training(NumOfTrainingLoop, TrainingRate)



//request for answer
NN_Construction(1, 0)
alert(Sig(NN))

