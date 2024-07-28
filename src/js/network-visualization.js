export default function displayModelArchitecture(model) {
  const table = document.createElement('table')
  table.style.width = '100%'
  table.style.borderCollapse = 'collapse'

  // Create table header
  const header = table.createTHead()
  const headerRow = header.insertRow()
  ;['Layer Type', 'Output Shape', 'Param #'].forEach((text) => {
    const th = document.createElement('th')
    th.textContent = text
    th.style.border = '1px solid #ddd'
    th.style.padding = '8px'
    th.style.backgroundColor = '#f2f2f2'
    headerRow.appendChild(th)
  })

  // Create table body
  const tbody = table.createTBody()
  model.layers.forEach((layer) => {
    const row = tbody.insertRow()
    const cells = [
      `${layer.name} (${layer.getClassName()})`,
      JSON.stringify(layer.outputShape),
      layer.countParams(),
    ]
    cells.forEach((text) => {
      const cell = row.insertCell()
      cell.textContent = text
      cell.style.border = '1px solid #ddd'
      cell.style.padding = '8px'
    })
  })

  // Add total parameters row
  const totalRow = tbody.insertRow()
  const totalNameCell = totalRow.insertCell()
  totalNameCell.textContent = 'Total params:'
  totalNameCell.style.fontWeight = 'bold'
  totalNameCell.style.border = '1px solid #ddd'
  totalNameCell.style.padding = '8px'

  const totalParamsCell = totalRow.insertCell()
  totalParamsCell.textContent = model.countParams().toLocaleString()
  totalParamsCell.style.fontWeight = 'bold'
  totalParamsCell.colSpan = 2
  totalParamsCell.style.border = '1px solid #ddd'
  totalParamsCell.style.padding = '8px'

  return table
}
