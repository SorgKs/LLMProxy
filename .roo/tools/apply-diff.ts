import { parametersSchema as z, defineCustomTool } from "@roo-code/types";
import * as fs from "fs/promises";
import * as path from "path";

/*
 * Очищает строку от невалидного JSON экранирования
 */
function sanitizeString(str: string): string {
  // Убираем лишние экранирования кавычек
  let result = str;
  
  // Если строка обёрнута в тройные кавычки
  if (result.startsWith('"""') && result.endsWith('"""')) {
    result = result.slice(3, -3);
  }
  
  // Заменяем \" на " (но не экранированные внутри JSON)
  result = result.replace(/\\"/g, '"');
  
  // Убираем экранирование переводов строк
  result = result.replace(/\\n/g, '\n');
  result = result.replace(/\\r/g, '\r');
  result = result.replace(/\\t/g, '\t');
  
  return result;
}

export default defineCustomTool({
  name: "apply_diff",
  description: `Применяет замену в файле, начиная поиск с указанной строки.
  
  Особенности:
  - Ищет ПЕРВОЕ вхождение search после указанной строки (включая её)
  - offset = номер строки (1-based), с которой начинать поиск
  - Если offset не указан или = 0, ищет с начала файла
  - Заменяет только первое найденное вхождение после offset`,

  // Прямые параметры, без обёртки nativeArgs
  parameters: z.object({
    filePath: z.string().describe("Путь к файлу (абсолютный или относительный)"),
    search: z.string().describe("Искомый текст для замены (точное совпадение)"),
    replace: z.string().describe("Текст, на который нужно заменить"),
    offset: z.number().optional().default(0).describe("Номер строки (1-based), с которой начинать поиск")
  }),

  async execute(args) {
    let { filePath, search, replace, offset = 0 } = args;
    
    // Очищаем строки от невалидного экранирования
    if (typeof search === 'string') {
      search = sanitizeString(search);
    }
    if (typeof replace === 'string') {
      replace = sanitizeString(replace);
    }
    
    // Нормализация пути
    const absolutePath = path.isAbsolute(filePath) 
      ? filePath 
      : path.resolve(process.cwd(), filePath);
    
    // Чтение файла
    const content = await fs.readFile(absolutePath, "utf-8");
    const lines = content.split("\n");
    
    // Проверка offset
    if (offset > lines.length) {
      throw new Error(`Offset ${offset} превышает количество строк в файле (${lines.length})`);
    }
    
    // Вычисляем позицию начала поиска в символах
    let startPos = 0;
    for (let i = 0; i < offset - 1 && i < lines.length; i++) {
      startPos += lines[i].length + 1; // +1 для \n
    }
    
    // Ищем первое вхождение
    const searchIndex = content.indexOf(search, startPos);
    
    if (searchIndex === -1) {
      const rangeMsg = offset > 0 ? `начиная со строки ${offset}` : "в файле";
      throw new Error(`"${search.substring(0, 100)}..." не найдено ${rangeMsg}`);
    }

    // Выполняем замену
    const newContent = 
      content.substring(0, searchIndex) + 
      replace + 
      content.substring(searchIndex + search.length);
    
    // Сохраняем
    await fs.writeFile(absolutePath, newContent, "utf-8");
    
    // Подсчёт строк
    const searchLines = search.split('\n').length;
    const replaceLines = replace.split('\n').length;
    const fileName = path.basename(filePath);
    
    return `Edit ${fileName} +${replaceLines} -${searchLines}`;
  }
});