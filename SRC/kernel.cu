

#include <stdio.h>
#include <algorithm>
#include <iterator>
#include <set>
#include <string>

#include <GL/glew.h>
#include <GL/wglew.h>
#include <GL/freeglut.h>
#include <SFML/System.hpp>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>


// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <curand.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>




#define NBody 16384
#define NB_THREAD 128

#define ScreenWidth 1024
#define ScreenHeight 768

#define invNBody (1.0f / NBody) //Variable de "vitesse de simulation" pour éviter une simulation explosive


__global__ void NBodyCUDA(float2 *nvBodyPos, float2 *nvBodyV, float2 *nvBodyDest, float tempInvNbody)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	//Step 3 Simuler l'interaction du body courant avec tous les autres bodies de nvBodyPos, ajouter à la vitesse, appliquer la vitesse
	
	float2 tempForce = { 0, 0 };
	
	for(int i = 0; i < NBody; i++) {

		float intGrav = 0;

		intGrav = -1.0f * (0.00001f + (nvBodyPos[index].x - nvBodyPos[i].x)*(nvBodyPos[index].x - nvBodyPos[i].x) + (nvBodyPos[index].y - nvBodyPos[i].y)*(nvBodyPos[index].y - nvBodyPos[i].y));
		tempForce.x += (nvBodyPos[index].x - nvBodyPos[i].x) / intGrav;
		tempForce.y += (nvBodyPos[index].y - nvBodyPos[i].y) / intGrav;

	}

	nvBodyV[index].x += tempForce.x * tempInvNbody;
	nvBodyV[index].y += tempForce.y * tempInvNbody;
	

	//Step 2 Copier la bonne donnée de nvBodyPos dans nvBodyDest
	nvBodyDest[index].x = nvBodyPos[index].x + nvBodyV[index].x;
	nvBodyDest[index].y = nvBodyPos[index].y + nvBodyV[index].y;

}



int main()
{
	//Allocation des tableaux de départ
	float2* BodyPos = (float2 *)malloc(NBody * sizeof(float2));
	float2* BodyV = (float2 *)malloc(NBody * sizeof(float2));

	//Allocation des tableaux de données initiales
	float2* nvBodyPos;
	cudaMalloc(&nvBodyPos, NBody * sizeof(float2));

	float2* nvBodyV;
	cudaMalloc(&nvBodyV, NBody * sizeof(float2));

	float2* nvBodyDest;
	cudaMalloc(&nvBodyDest, NBody * sizeof(float2));

	//Verification d'erreur
	if (nvBodyPos == NULL || nvBodyDest == NULL || nvBodyV == NULL)
	{
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	//position initiale aléatoire
	srand(time(NULL));
	for (int i = 0; i < NBody; i++)
	{
		BodyPos[i].x = (ScreenWidth / 2) + 600 * (-0.5 + (rand() / (float)RAND_MAX));
		BodyPos[i].y = (ScreenHeight / 2) + 300 * (-0.5 + (rand() / (float)RAND_MAX));
	}

	//Initialisation de SFML
	sf::VertexArray tempArray;
	tempArray.resize(NBody);
	tempArray.setPrimitiveType(sf::Points);
	sf::RenderWindow window(sf::VideoMode(1024, 768), "My window");

	int blockSize = NB_THREAD;
	int gridSize = (NBody + blockSize - 1) / blockSize;


	while (window.isOpen())
	{
		sf::Event event;
		while (window.pollEvent(event))
		{

		}
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape)) exit(0);

		//Step 1
		//Copie des positions vers le GPU
		cudaMemcpy(nvBodyPos, BodyPos, NBody * sizeof(float2), cudaMemcpyHostToDevice);

		//Lancement du kernel
		NBodyCUDA <<<gridSize, blockSize >>> (nvBodyPos, nvBodyV, nvBodyDest, invNBody);

		//Récupération des nouvelles positions
		cudaMemcpy(BodyPos, nvBodyDest, NBody * sizeof(float2), cudaMemcpyDeviceToHost);

		//Copie des données dans la Vertex Array pour l'affichage
		for (int i = 0; i < NBody; i++)
		{
			tempArray[i].position.x = BodyPos[i].x;
			tempArray[i].position.y = BodyPos[i].y;
		}


		//Affichage
		window.clear(sf::Color::Black);
		window.draw(tempArray);
		window.display();
		//exit(0);
	}
}

