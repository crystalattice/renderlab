// REFERENCE ONLY. Nothing in this file has been executed.
// Run only after explicit approval for both uploads and Cloud GPU spend.
// First invoke these exact two tools, then execute each returned single-use PUT
// command and retain the actual `name` from each successful upload response.
// await tools.mcp__comfy_cloud__upload_file({
//   client_os: "linux",
//   file_path: "/home/codyjackson/Downloads/img_00562__firered-image-edit-1.1__single-pass__621146938678618_00001_.png"
// });
// await tools.mcp__comfy_cloud__upload_file({
//   client_os: "linux",
//   file_path: "/home/codyjackson/PycharmProjects/renderlab/input/renderlab_afccbc5cc07c4f88b8f397e2104664f5.png"
// });

// uploadTarget and uploadIdentity are the parsed successful upload response
// objects, not guessed basenames or local paths. This function submits one job.
async function executeAfterApproval(uploadTarget, uploadIdentity) {
  if (!uploadTarget.name || !uploadIdentity.name || uploadTarget.name === uploadIdentity.name) {
    throw new Error("Two distinct verified Cloud input names are required");
  }
  const result = await tools.exec_command({
    cmd: "cat /home/codyjackson/PycharmProjects/renderlab/renderlab/experiments/qwen_face_swap_clothed_v1/api.json",
    max_output_tokens: 6000
  });
  if (result.exit_code !== 0) throw new Error("Could not read the validated local API payload");
  const workflow = JSON.parse(result.output);
  workflow["470"].inputs.image = uploadTarget.name;
  workflow["471"].inputs.image = uploadIdentity.name;
  return await tools.mcp__comfy_cloud__submit_workflow({workflow, confirm: true});
}
