#!/bin/bash

#allData=("RobTetley" "AlejandraGuzman" "RiciBarrientos/NubG4-UASmyrGFP_Control" "RiciBarrientos/NubG4-UASmyrGFP-UASMbsRNAi" "RiciBarrientos/NubG4-UASmyrGFP-UASRokRNAi" "RiciBarrientos/CLS1" "RiciBarrientos/UpsideDown_CorrectedPhotobleaching" "RiciBarrientos/CellCycle")
learningRateParams=( "0.0001" "0.0001") # different parameter per method
weightDecayParams=( "0.0001" "0.0001") # different parameter per method

f_maps=( "32" "32") # different parameter per method
stride_shape=("[15,80,80]" "[36,100,100]")
patch_shape=("[36,400,400]" "[36,400,400]")

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo $DIR

len=${#learningRateParams[@]}

for (( numData=0; numData<$len; numData++ ))
do
	echo "#Learning rate ${learningRateParams[numData]} - Weight Decay ${weightDecayParams[numData]}"

	weightDecay=${weightDecayParams[numData]}
	learningRate=${learningRateParams[numData]}

	# Instance segmentation
	List="BCEDiceLoss"
	arrayMethods=($List)

	for numMethod in {0..0}
	do 
		echo "##${arrayMethods[numMethod]}"
		if [ ${weightDecay[numMethod]} != -1 ]; then
			
			newCurrentPath="${DIR}/results/LR_${learningRate[numMethod]}_WD_${weightDecay[numMethod]}_fm_${f_maps[numMethod]}_ps_${patch_shape[numMethod]}_ss_${stride_shape[numMethod]}"
			echo $newCurrentPath
			mkdir -p "$newCurrentPath/outputPredicted/"

			if [ ! -d "$newCurrentPath/models" ]; then

				cp "$DIR/initialModel.pytorch" "$newCurrentPath/initialModel.pytorch"
				
				sed -e "s@currentPath@${newCurrentPath}@g" \
				 -e "s@weightDecay@${weightDecay[numMethod]}@g" \
				 -e "s@learningRate@${learningRate[numMethod]}@g" \
				 -e "s@FMAPS@${f_maps[numMethod]}@g" \
				 -e "s@STRIDESHAPE@${stride_shape[numMethod]}@g" \
				 -e "s@PATCHSHAPE@${patch_shape[numMethod]}@g" \
				 $DIR/trainingModel_generic.yaml > $newCurrentPath/training_settings.yaml

				echo "### Training model"
				train3dunet --config $newCurrentPath/training_settings.yaml > $newCurrentPath/out_training.log

				sed -e "s@currentPath@${newCurrentPath}@g" \
				 -e "s@FMAPS@${f_maps[numMethod]}@g" \
				 -e "s@STRIDESHAPE@${stride_shape[numMethod]}@g" \
				 -e "s@PATCHSHAPE@${patch_shape[numMethod]}@g" \
				 $DIR/predictionModel_generic.yaml > $newCurrentPath/test_settings.yaml

				echo "### Predicting from trained model"
				predict3dunet --config $newCurrentPath/test_settings.yaml > $newCurrentPath/out_prediction.log
			fi
		fi
		echo "##${arrayMethods[numMethod]} - Done!"
	done
	echo "#${allData[numData]} - Done!"
done
