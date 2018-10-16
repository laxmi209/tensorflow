import { Component,OnInit  } from '@angular/core';
import * as tf from '@tensorflow/tfjs'
import { loadFrozenModel, FrozenModel  } from '@tensorflow/tfjs-converter'
import * as tfc from '@tensorflow/tfjs-core';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
	title = 'tensormobilenetv2';
	labels = ["daisy","dandelion","roses","sunflowers","tulips"];
	MODEL_URL = './assets/tensorflowjs_model.pb';
	WEIGHTS_URL = './assets/weights_manifest.json';
	IMAGE_SIZE = 96; // Model input size
	model: FrozenModel;
	timemy: string;
	imagedaisy = new Image();
	mypredictions=[];
	mypredicttime ='';
	myoutt;
	ngOnInit() {
		loadFrozenModel(this.MODEL_URL, this.WEIGHTS_URL).then(loadedmodel =>{
			// Warm up GPU
			const t0 = performance.now()
			const input = tfc.zeros([1, this.IMAGE_SIZE, this.IMAGE_SIZE, 3])
			//model.predict({ input }) // MobileNet V1
			loadedmodel.predict({ Placeholder: input }) // MobileNet V2
			this.model = loadedmodel;
			this.timemy = `${(performance.now() - t0).toFixed(1)} ms`
			console.log("Model loaded in :", this.timemy);
			this.imagedaisy.src = './assets/checkdaisy.jpg';
			
		});
	}
	mypredict(){
		  this.predict(this.imagedaisy,this.model).then((mine)=>{
    //this.outputjson = mine.predictions;
    this.mypredictions = JSON.parse(JSON.stringify(mine.predictions));
    this.mypredicttime = JSON.parse(JSON.stringify(mine.time));
  });
	}
	predict = async (img, model) => {
		const t0 = performance.now()
		const image = tf.fromPixels(img).toFloat()
		const resized = tf.image.resizeBilinear(image, [this.IMAGE_SIZE, this.IMAGE_SIZE])
		const offset = tf.scalar(255 / 2)
		const normalized = resized.sub(offset).div(offset)
		const input = normalized.expandDims(0)
		//const output = await tf.tidy(() => model.predict({ input })).data() // MobileNet V1
		 const output = await tf.tidy(() => model.predict({ Placeholder: input })).data() // MobileNet V2
		const predictions = this.labels
		.map((label, index) => ({ label, accuracy: output[index] }))
		.sort((a, b) => b.accuracy - a.accuracy)
		const time = `${(performance.now() - t0).toFixed(1)} ms`

		return { predictions, time }
	}

	mypredict1(){
	  const t0 = performance.now();
	  const image = tfc.fromPixels(this.imagedaisy);
	  this.myoutt= this.predict1(image);
	  const myoutput = this.getTopKClasses1(this.myoutt, 3);
	  this.mypredictions = JSON.parse(JSON.stringify(myoutput));
	  const time = `${(performance.now() - t0).toFixed(1)} ms`;
	  this.mypredicttime = JSON.parse(JSON.stringify(time));
	  //this.myoutt.dispose();
	}

	predict1(input: tfc.Tensor): tfc.Tensor1D {
	  const PREPROCESS_DIVISOR = tfc.scalar(255 / 2);
	  type TensorMap = {[name: string]: tfc.Tensor};
	  const image = tfc.fromPixels(this.imagedaisy).toFloat();
	  const resized = tfc.image.resizeBilinear(image, [this.IMAGE_SIZE, this.IMAGE_SIZE]);
	  const INPUT_NODE_NAME = 'Placeholder';
	  const OUTPUT_NODE_NAME = 'final_result';
	  const preprocessedInput = tfc.div(
	    tfc.sub(resized, PREPROCESS_DIVISOR),
	    PREPROCESS_DIVISOR);
	  const reshapedInput =
	    preprocessedInput.reshape([1, ...preprocessedInput.shape]);
	  const dict: TensorMap = {};
	  dict[INPUT_NODE_NAME] = reshapedInput;
	  return this.model.predict(dict) as tfc.Tensor1D;
	  }

	getTopKClasses1(predictions: tfc.Tensor1D, topK: number) {
	    const values = predictions.dataSync();
	    predictions.dispose();

	    let predictionList = [];
	    for (let i = 0; i < values.length; i++) {
	      predictionList.push({value: values[i], index: i});
	    }
	    predictionList = predictionList.sort((a, b) => {
	    return b.value - a.value;
	    }).slice(0, topK);

	    return predictionList.map(x => {
	      return {label: this.labels[x.index], accuracy: x.value};
	    });
	}

}

/*
import * as tf from '@tensorflow/tfjs'
import { loadFrozenModel } from '@tensorflow/tfjs-converter'
import labels from './labels.json'

const ASSETS_URL = `${window.location.origin}/assets`
const MODEL_URL = `${ASSETS_URL}/model/tensorflowjs_model.pb`
const WEIGHTS_URL = `${ASSETS_URL}/model/weights_manifest.json`
const IMAGE_SIZE = 128 // Model input size

const loadModel = async () => {
  const model = await loadFrozenModel(MODEL_URL, WEIGHTS_URL)
  // Warm up GPU
  const input = tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])
  model.predict({ input }) // MobileNet V1
  // model.predict({ Placeholder: input }) // MobileNet V2
  return model
}

const predict = async (img, model) => {
  const t0 = performance.now()
  const image = tf.fromPixels(img).toFloat()
  const resized = tf.image.resizeBilinear(image, [IMAGE_SIZE, IMAGE_SIZE])
  const offset = tf.scalar(255 / 2)
  const normalized = resized.sub(offset).div(offset)
  const input = normalized.expandDims(0)
  const output = await tf.tidy(() => model.predict({ input })).data() // MobileNet V1
  // const output = await tf.tidy(() => model.predict({ Placeholder: input })).data() // MobileNet V2
  const predictions = labels
    .map((label, index) => ({ label, accuracy: output[index] }))
    .sort((a, b) => b.accuracy - a.accuracy)
  const time = `${(performance.now() - t0).toFixed(1)} ms`
  return { predictions, time }
}

const start = async () => {
  const input = document.getElementById('input')
  const output = document.getElementById('output')
  const model = await loadModel()
  const predictions = await predict(input, model)
  output.append(JSON.stringify(predictions, null, 2))
}

start()
*/