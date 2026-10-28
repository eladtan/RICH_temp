/*! \file LinearGaussArrays.hpp
  \brief The flat arrays LinearGauss3D computes in, kept between calls. The flux kernels read its face values.
*/
#ifndef LINEAR_GAUSS_ARRAYS_HPP
#define LINEAR_GAUSS_ARRAYS_HPP 1

#include <cstddef>
#include <cstdint>
#include "utils/Portable.hpp"
#include "LinearGaussKernels.hpp"

// The flat arrays the kernels use (see LinearGaussKernels.hpp), kept between calls
struct LinearGaussArrays
{
	gauss::Config config;
	std::size_t cellsNum = 0;
	std::size_t pointsNum = 0;
	std::size_t facesNum = 0;
	std::size_t wallsNum = 0; // box faces with a ghost
	portable::Buffer<std::size_t> cellFaceOffsets, cellFaces, faceNeighbors, stateIndex, wallSlot;
	portable::Buffer<double> faceCM, faceArea, meshPoint, pointCM, volume, width;
	portable::Buffer<double> states;
	portable::Buffer<std::uint8_t> skip;
	portable::Buffer<double> soundSpeeds, slopes, naive, wallSlopes, faceValues;
	portable::Buffer<std::uint8_t> soundSpeedStatus, cellStatus, faceStatus;
	bool inputsOnDevice = false;
	std::size_t slopesOnDevice = 0; // points whose slopes the device array holds
	bool naivePending = false; // the unlimited slopes are not yet in naive_rslopes_ (GetSlopesUnlimited unpacks them)
	bool naiveOnDevice = false; // and are still on the device
	portable::Where faceValuesWhere = portable::Where::Host; // where the last face values were computed
	bool sr = false; // the face values' velocities are the special-relativistic u = gamma v

	gauss::Mesh MeshView(portable::Where where)
	{
		gauss::Mesh mesh;
		mesh.cellFaceOffsets = this->cellFaceOffsets.Data(where);
		mesh.cellFaces = this->cellFaces.Data(where);
		mesh.faceNeighbors = this->faceNeighbors.Data(where);
		mesh.faceCM = this->faceCM.Data(where);
		mesh.faceArea = this->faceArea.Data(where);
		mesh.wallSlot = this->wallSlot.Data(where);
		mesh.meshPoint = this->meshPoint.Data(where);
		mesh.pointCM = this->pointCM.Data(where);
		mesh.volume = this->volume.Data(where);
		mesh.width = this->width.Data(where);
		mesh.stateIndex = this->stateIndex.Data(where);
		mesh.cellsNum = this->cellsNum;
		mesh.pointsNum = this->pointsNum;
		return mesh;
	}

	gauss::State StateView(portable::Where where)
	{
		gauss::State state;
		state.values = this->states.Data(where);
		state.skip = this->skip.Data(where);
		return state;
	}

	void InputsToDevice(void)
	{
		for(portable::Buffer<std::size_t> *buffer : {&this->cellFaceOffsets, &this->cellFaces, &this->faceNeighbors, &this->stateIndex, &this->wallSlot})
		{
			buffer->ToDevice();
		}
		for(portable::Buffer<double> *buffer : {&this->faceCM, &this->faceArea, &this->meshPoint, &this->pointCM, &this->volume,
			&this->width, &this->states})
		{
			buffer->ToDevice();
		}
		this->skip.ToDevice();
		this->inputsOnDevice = true;
	}
};

#endif // LINEAR_GAUSS_ARRAYS_HPP
