import { CanvasInput } from "../components/CanvasInput";
import { MLPGraph } from "../components/MLPGraph";
import { ModelConfigurator } from "../components/ModelConfigurator";
import { ModelStoragePanel } from "../components/ModelStoragePanel";
import { PredictPanel } from "../components/PredictPanel";
import { TrainingChart } from "../components/TrainingChart";
import { TrainingPanel } from "../components/TrainingPanel";

export default function Board() {
  return (
    <div className="flex h-[40em] flex-col">
      <div className="flex flex-1">
        {/* Left Column */}
        <div className="flex w-[20em] flex-col">
          <ModelStoragePanel />
          <ModelConfigurator />
        </div>

        {/* Center Column */}
        <div className="flex flex-grow flex-col">
          <div className="flex justify-start">
            <CanvasInput />
            <div className="h-full flex-grow">
              <PredictPanel />
            </div>
          </div>
          <div className="flex-grow">
            <MLPGraph />
          </div>
        </div>

        {/* Right Column */}
        <div className="flex w-[20em] flex-col">
          <TrainingPanel />
          <TrainingChart />
        </div>
      </div>
    </div>
  );
}
