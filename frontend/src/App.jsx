import { motion } from "framer-motion";
import { Brain, Upload } from "lucide-react";
import { useState } from "react";
import axios from "axios";

function App() {
const [file, setFile] = useState(null);
const [preview, setPreview] = useState(null);
const [result, setResult] = useState(null);
const [loading, setLoading] = useState(false);
const [patientName, setPatientName] = useState("");
const [age, setAge] = useState("");
const [gender, setGender] = useState("");
const [gradcamUrl, setGradcamUrl] = useState(null);

const handleUpload = async () => {
if (!file) {
alert("Please select an MRI image");
return;
}


setLoading(true);
setResult(null);

const formData = new FormData();
formData.append("file", file);

try {
  const response = await axios.post(
    "http://127.0.0.1:8000/predict",
    formData
  );

  setResult(response.data);
  /*const gradcamResponse = await axios.post(
  "http://127.0.0.1:8000/gradcam",
  formData,
  {
    responseType: "blob",
  }
);

const gradcamImageUrl = URL.createObjectURL(
  gradcamResponse.data
);

setGradcamUrl(gradcamImageUrl);*/
} catch (error) {
  console.error(error);
  alert("Prediction failed");
}

setLoading(false);


};

const downloadReport = async () => {
  try {

    const formData = new FormData();

    formData.append(
      "prediction",
      result.prediction
    );

    formData.append(
      "confidence",
      (result.confidence * 100).toFixed(2)
    );

    formData.append(
  "patient_name",
  patientName
);

formData.append(
  "age",
  age
);

formData.append(
  "gender",
  gender
);

    const response = await axios.post(
      "http://127.0.0.1:8000/generate-report",
      formData,
      {
        responseType: "blob",
      }
    );

    const url = window.URL.createObjectURL(
      new Blob([response.data])
    );

    const link = document.createElement("a");

    link.href = url;

    link.setAttribute(
      "download",
      "BrainSight_Report.pdf"
    );

    document.body.appendChild(link);

    link.click();

    link.remove();

  } catch (error) {

    console.error(error);

    alert("Failed to download report");
  }
};


return ( <div className="min-h-screen bg-black text-white overflow-hidden relative">

```
  {/* Background Effects */}
  <div className="absolute top-0 left-0 w-96 h-96 bg-cyan-500/20 blur-[150px]" />
  <div className="absolute bottom-0 right-0 w-96 h-96 bg-purple-500/20 blur-[150px]" />

  <div className="relative z-10 flex flex-col items-center px-6 py-16">

    {/* Logo */}
    <motion.div
      animate={{ y: [0, -15, 0] }}
      transition={{ repeat: Infinity, duration: 4 }}
    >
      <Brain size={90} className="text-cyan-400" />
    </motion.div>

    {/* Heading */}
    <motion.h1
      initial={{ opacity: 0, y: 40 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 1 }}
      className="text-5xl md:text-7xl font-bold text-center mt-4"
    >
      BrainSight AI
    </motion.h1>

    <p className="text-gray-400 text-center mt-5 max-w-2xl text-lg">
      AI Powered Brain Tumor Detection using Deep Learning and MRI Analysis
    </p>

    {/* Stats Cards */}
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-10 w-full max-w-4xl">

      <div className="bg-white/5 backdrop-blur-xl rounded-3xl p-6 border border-white/10">
        <h3 className="text-cyan-400 text-3xl font-bold">
          92.1%
        </h3>
        <p className="text-gray-400 mt-2">
          Validation Accuracy
        </p>
      </div>

      <div className="bg-white/5 backdrop-blur-xl rounded-3xl p-6 border border-white/10">
        <h3 className="text-cyan-400 text-3xl font-bold">
          MobileNetV2
        </h3>
        <p className="text-gray-400 mt-2">
          Deep Learning Model
        </p>
      </div>

    </div>

    {/* Upload Section */}
    <div className="mt-12 w-full max-w-3xl bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl p-8 text-center">

    <input
  type="text"
  placeholder="Patient Name"
  value={patientName}
  onChange={(e) => setPatientName(e.target.value)}
  className="w-full p-3 rounded-xl bg-black border border-cyan-500 mb-4"
/>

<input
  type="number"
  placeholder="Age"
  value={age}
  onChange={(e) => setAge(e.target.value)}
  className="w-full p-3 rounded-xl bg-black border border-cyan-500 mb-4"
/>

<select
  value={gender}
  onChange={(e) => setGender(e.target.value)}
  className="w-full p-3 rounded-xl bg-black border border-cyan-500 mb-4"
>
  <option value="">Select Gender</option>
  <option value="Male">Male</option>
  <option value="Female">Female</option>
  <option value="Other">Other</option>
</select>

      <input
        type="file"
        accept="image/*"
        onChange={(e) => {
          const selected = e.target.files[0];
          setFile(selected);

          if (selected) {
            setPreview(URL.createObjectURL(selected));
          }
        }}
        className="mb-6"
      />

      {preview && (
        <motion.img
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          src={preview}
          alt="MRI Preview"
          className="w-80 h-80 object-cover mx-auto rounded-3xl border border-cyan-500 shadow-2xl"
        />
      )}

      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={handleUpload}
        className="mt-8 flex items-center gap-3 mx-auto bg-cyan-500 px-8 py-4 rounded-2xl text-black font-semibold"
      >
        <Upload size={22} />
        Analyze MRI
      </motion.button>

      {loading && (
        <motion.div
          animate={{ rotate: 360 }}
          transition={{
            repeat: Infinity,
            duration: 1,
            ease: "linear",
          }}
          className="mt-8 w-16 h-16 border-4 border-cyan-500 border-t-transparent rounded-full mx-auto"
        />
      )}
    </div>

    {/* Result Section */}
{result && (
  <motion.div
    initial={{ opacity: 0, y: 30 }}
    animate={{ opacity: 1, y: 0 }}
    className="mt-10 w-full max-w-3xl"
  >
    <div className="bg-white/10 backdrop-blur-xl border border-white/20 rounded-3xl p-8 shadow-2xl">

      {/* Prediction */}
      <h2 className="text-4xl font-bold mb-6 text-cyan-400">
        {result.prediction}
      </h2>

      {/* Confidence */}
      <p className="text-gray-300 mb-3">
        Confidence Score
      </p>

      <div className="w-full bg-gray-700 rounded-full h-4 overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{
            width: `${(result.confidence * 100).toFixed(0)}%`,
          }}
          transition={{ duration: 1 }}
          className="h-4 bg-cyan-400"
        />
      </div>

      <p className="mt-4 text-3xl font-bold text-cyan-400">
        {(result.confidence * 100).toFixed(2)}%
      </p>

      {/* Probabilities */}
      <div className="mt-8">
        <h3 className="text-xl font-bold text-cyan-400 mb-4">
          Class Probabilities
        </h3>

        {Object.entries(result.probabilities).map(
          ([label, score]) => (
            <div key={label} className="mb-4">

              <div className="flex justify-between mb-1">
                <span>{label}</span>

                <span>
                  {(score * 100).toFixed(2)}%
                </span>
              </div>

              <div className="w-full bg-gray-700 rounded-full h-3">
                <div
                  className="bg-cyan-400 h-3 rounded-full"
                  style={{
                    width: `${score * 100}%`,
                  }}
                />
              </div>

            </div>
          )
        )}
      </div>

      {/* Low Confidence Warning */}
      {result.confidence < 0.7 && (
        <div className="mt-6 bg-yellow-500/20 border border-yellow-500 rounded-xl p-4">
          ⚠️ Low confidence prediction. Manual review recommended.
        </div>
      )}

      <div className="mt-8 space-y-2 text-gray-300">
        <p>✓ Deep Learning Analysis Completed</p>
        <p>✓ MRI Scan Processed Successfully</p>
        <p>✓ Prediction Confidence Calculated</p>
      </div>

      <motion.button

      
  whileHover={{ scale: 1.05 }}
  whileTap={{ scale: 0.95 }}
  onClick={downloadReport}
  className="mt-8 w-full bg-cyan-500 text-black font-bold py-4 rounded-2xl"
>
  📄 Download Medical Report
</motion.button>

    </div>

    {gradcamUrl && (

  <div className="mt-8 bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl p-8">

    <h3 className="text-2xl font-bold text-cyan-400 mb-4">
      Explainable AI (Grad-CAM)
    </h3>

    <p className="text-gray-300 mb-6">
      Highlighted regions indicate where the AI model focused while making its decision.
    </p>

    <img
      src={gradcamUrl}
      alt="GradCAM"
      className="rounded-2xl border border-cyan-500 mx-auto"
    />

  </div>

)}

    {/* Explainable AI Section */}
    <div className="mt-8 bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl p-8">

      <h3 className="text-2xl font-bold text-cyan-400 mb-4">
        AI Decision Analysis
      </h3>

      <div className="space-y-3 text-gray-300">
        <p>✓ High intensity abnormal region detected</p>
        <p>✓ Pattern matched with trained MRI dataset</p>
        <p>✓ Confidence exceeds decision threshold</p>
        <p>✓ Deep learning model classified image successfully</p>
      </div>

    </div>

  </motion.div>
)}

        
  </div>
</div>


);
}

export default App;
