import React, { useMemo } from "react";
import * as tf from "@tensorflow/tfjs";
import { useMLPStore } from "../stores/useMLPStore";

type LayerParams = {
  name: string;
  inputSize: number;
  outputSize: number;
  weights: number[][];
  biases: number[];
};

const formatNumber = (v: number) => v.toFixed(4);

const extractLayers = (model: tf.LayersModel): LayerParams[] => {
  const result: LayerParams[] = [];
  for (const layer of model.layers) {
    const [kernel, bias] = layer.getWeights();
    if (!kernel) continue;
    const weights = kernel.arraySync() as number[][];
    const biases = bias
      ? (bias.dataSync() as Float32Array)
      : new Float32Array();
    result.push({
      name: layer.name,
      inputSize: weights.length,
      outputSize: weights[0]?.length ?? 0,
      weights,
      biases: Array.from(biases),
    });
  }
  return result;
};

export const ModelParametersPanel: React.FC = () => {
  const model = useMLPStore((s) => s.model);
  const historyLength = useMLPStore((s) => s.trainingHistory.epochs.length);

  const layers = useMemo(
    () => (model ? extractLayers(model) : []),
    // historyLength: les poids changent pendant l'entraînement
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [model, historyLength],
  );

  const totalParams = model ? model.countParams() : 0;

  return (
    <div className="flex w-full flex-col gap-2 bg-white p-2">
      <h2 className="px-2 text-lg font-bold">
        Paramètres du modèle ({totalParams} au total)
      </h2>

      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="border-b text-left">
            <th className="px-2 py-1">Couche</th>
            <th className="px-2 py-1">Entrée → Sortie</th>
            <th className="px-2 py-1">Poids</th>
            <th className="px-2 py-1">Biais</th>
            <th className="px-2 py-1">Paramètres</th>
          </tr>
        </thead>
        <tbody>
          {layers.map((layer) => {
            const nbWeights = layer.inputSize * layer.outputSize;
            const nbBiases = layer.biases.length;
            return (
              <tr key={layer.name} className="border-b align-top">
                <td className="px-2 py-1 font-mono">{layer.name}</td>
                <td className="px-2 py-1">
                  {layer.inputSize} → {layer.outputSize}
                </td>
                <td className="px-2 py-1">{nbWeights}</td>
                <td className="px-2 py-1">{nbBiases}</td>
                <td className="px-2 py-1">{nbWeights + nbBiases}</td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <div className="flex flex-wrap gap-4 px-2">
        {layers.map((layer) => {
          const flat = layer.weights.flat();
          const min = flat.length ? Math.min(...flat) : 0;
          const max = flat.length ? Math.max(...flat) : 0;
          return (
            <details
              key={layer.name}
              className="min-w-[16em] rounded border p-2 text-sm"
            >
              <summary className="cursor-pointer font-mono">
                {layer.name}
              </summary>
              <div className="mt-2 space-y-1">
                <p>
                  Poids : min {formatNumber(min)} / max {formatNumber(max)}
                </p>
                <p className="font-semibold">Biais</p>
                <div className="max-h-48 overflow-auto font-mono text-xs">
                  {layer.biases.map((b, i) => (
                    <div key={i}>
                      [{i}] {formatNumber(b)}
                    </div>
                  ))}
                </div>
              </div>
            </details>
          );
        })}
      </div>
    </div>
  );
};
